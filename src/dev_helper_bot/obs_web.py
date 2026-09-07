"""In-process веб-дашборд телеметрии (change add-obs-web-dashboard, design D1/D4).

aiohttp-приложение, поднимаемое в том же процессе, что и long-polling бота,
на loopback (`127.0.0.1:8765` по умолчанию). Отдаёт HTML+CSS-страницы
(`/`, `/runs`, `/runs/{id}`, `/audit`) и JSON `/api/*` с query-фильтрами
`label`/`since`. Клиентский JS опрашивает `/api/*` раз в ~1.5 с и
перерисовывает блоки, не сбрасывая выбранный `run_id` (он в URL).

Если хранилище телеметрии недоступно (store is None), все страницы отдают
понятную заглушку «телеметрия недоступна» (design D2, спека obs-web-dashboard).
Запись телеметрии и обработка сообщений бота при этом не страдают.

Lifecycle (design D1): старт AppRunner/TCPSite до polling, stop в finally.
Сбой bind (занятый порт) — исключение до polling с понятной причиной.
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from aiohttp import web

from dev_helper_bot.telemetry import (
    DEFAULT_AUDIT_CACHED_PRICE_MULTIPLIER,
    DEFAULT_AUDIT_PRICE_INPUT_PER_M,
    DEFAULT_AUDIT_PRICE_OUTPUT_PER_M,
    TelemetryStore,
)
from dev_helper_bot.obs_web_pages import (
    PRICES_KEY,
    STORE_KEY,
    handle_audit_page,
    handle_run_page,
    handle_runs_page,
    handle_summary_page,
)

log = logging.getLogger(__name__)

DEFAULT_OBS_WEB_PORT = 8765
"""Дефолтный порт дашборда (design D6); переопределяется OBS_WEB_PORT."""

DEFAULT_BIND_HOST = "127.0.0.1"
"""Bind всегда loopback (design D6): меньше шансов случайно открыть LAN."""

POLL_INTERVAL_MS = 1500
"""Интервал клиентского опроса (design D4, спека «Обновление опросом»): ~1.5 с."""


def build_app(
    store: TelemetryStore | None,
    *,
    price_input_per_m: float = DEFAULT_AUDIT_PRICE_INPUT_PER_M,
    price_output_per_m: float = DEFAULT_AUDIT_PRICE_OUTPUT_PER_M,
    cached_price_multiplier: float = DEFAULT_AUDIT_CACHED_PRICE_MULTIPLIER,
) -> web.Application:
    """Собирает aiohttp-app с маршрутами HTML-страниц и JSON /api/*.

    store=None — режим «телеметрия недоступна»: хендлеры отдают заглушку
    (design D2). Цены передаются в контекст app для аудита (design D5).
    """
    app = web.Application()
    app[STORE_KEY] = store
    app[PRICES_KEY] = {
        "price_input_per_m": price_input_per_m,
        "price_output_per_m": price_output_per_m,
        "cached_price_multiplier": cached_price_multiplier,
    }
    app.add_routes([
        web.get("/", handle_summary_page),
        web.get("/runs", handle_runs_page),
        web.get("/runs/{run_id}", handle_run_page),
        web.get("/audit", handle_audit_page),
        web.get("/api/summary", handle_api_summary),
        web.get("/api/runs", handle_api_runs),
        web.get("/api/runs/{run_id}", handle_api_run),
        web.get("/api/audit", handle_api_audit),
    ])
    return app


def _store(request: web.Request) -> TelemetryStore | None:
    return request.app[STORE_KEY]


def _prices(request: web.Request) -> dict[str, float]:
    return request.app[PRICES_KEY]


def _filters(request: web.Request) -> tuple[str | None, str | None]:
    """Читает query-фильтры label/since (пустая строка → None)."""
    label = request.query.get("label") or None
    since = request.query.get("since") or None
    return label, since


def _dataclass_to_dict(obj: Any) -> Any:
    """Рекурсивно конвертирует dataclass (в т.ч. вложенные списки/словари) в
    JSON-сериализуемый вид. None сохраняется (absence ≠ 0)."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


# --- JSON /api/* хендлеры (design D4) ---


async def handle_api_summary(request: web.Request) -> web.Response:
    store = _store(request)
    if store is None:
        return web.json_response({"available": False})
    label, since = _filters(request)
    summary = await store.read_summary(label, since)
    return web.json_response({"available": True, "summary": _dataclass_to_dict(summary)})


async def handle_api_runs(request: web.Request) -> web.Response:
    store = _store(request)
    if store is None:
        return web.json_response({"available": False, "runs": []})
    label, since = _filters(request)
    runs = await store.read_runs(label, since)
    return web.json_response({
        "available": True,
        "runs": [_dataclass_to_dict(r) for r in runs],
    })


async def handle_api_run(request: web.Request) -> web.Response:
    store = _store(request)
    if store is None:
        return web.json_response({"available": False})
    try:
        run_id = int(request.match_info["run_id"])
    except ValueError:
        return web.json_response({"available": True, "run": None}, status=404)
    detail = await store.read_run(run_id)
    return web.json_response({"available": True, "run": _dataclass_to_dict(detail)})


async def handle_api_audit(request: web.Request) -> web.Response:
    store = _store(request)
    if store is None:
        return web.json_response({"available": False})
    label, since = _filters(request)
    prices = _prices(request)
    audit = await store.read_audit(
        label, since,
        price_input_per_m=prices["price_input_per_m"],
        price_output_per_m=prices["price_output_per_m"],
        cached_price_multiplier=prices["cached_price_multiplier"],
    )
    return web.json_response({"available": True, "audit": _dataclass_to_dict(audit)})


# --- Lifecycle (design D1): AppRunner/TCPSite, fail-fast на порте ---


async def start_app(
    app: web.Application,
    *,
    host: str = DEFAULT_BIND_HOST,
    port: int = DEFAULT_OBS_WEB_PORT,
) -> web.AppRunner:
    """Поднимает AppRunner + TCPSite на host:port; при сбое bind пробрасывает
    исключение (design D2: занятый порт → процесс бота не стартует polling)."""
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    try:
        await site.start()
    except OSError as exc:
        await runner.cleanup()
        raise OSError(
            f"Не удалось занять порт дашборда {host}:{port} — {exc.strerror} "
            f"(errno {exc.errno}). Освободите порт или задайте OBS_WEB_PORT."
        ) from exc
    log.info("obs web dashboard: http://%s:%d/", host, port)
    return runner


async def stop_app(runner: web.AppRunner | None) -> None:
    """Корректно останавливает AppRunner (вызывается в finally main)."""
    if runner is not None:
        await runner.cleanup()
