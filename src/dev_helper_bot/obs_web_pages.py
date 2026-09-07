"""HTML+CSS-страницы и клиентский JS дашборда (change add-obs-web-dashboard).

Вынесены из obs_web.py, чтобы держать хендлеры и шаблоны отдельно от
lifecycle/JSON. Страницы — HTML-оболочки с встроенным CSS и JS-поллом
/api/* раз в ~1.5 с (design D4). Выбранный run_id в URL не сбрасывается
опросом (task 2.2). Store=None → заглушка «телеметрия недоступна» (task 2.3).
"""
from __future__ import annotations

from aiohttp import web

POLL_INTERVAL_MS = 1500
"""Интервал клиентского опроса (design D4): ~1.5 с. Дублируется из obs_web,
чтобы избежать циклического импорта — obs_web импортирует хендлеры отсюда."""

STORE_KEY = web.AppKey("obs_store", object)
"""Ключ контекста app, под которым лежит TelemetryStore (или None).
Определён здесь (а не в obs_web), т.к. obs_web импортирует хендлеры —
AppKey должен жить в импортируемом модуле, чтобы оба видели один ключ."""

PRICES_KEY = web.AppKey("obs_prices", dict)
"""Ключ контекста app с ценами для аудита (design D5)."""

_CSS = """
* { box-sizing: border-box; }
body { font: 14px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif;
       margin: 0; color: #1a1a1a; background: #fafafa; }
header { background: #1f2937; color: #fff; padding: 10px 18px; }
header a { color: #e5e7eb; text-decoration: none; margin-right: 14px; }
header a.active, header a:hover { color: #fff; }
main { max-width: 1000px; margin: 16px auto; padding: 0 16px; }
.filters { display: flex; gap: 10px; align-items: center; margin-bottom: 14px;
           flex-wrap: wrap; }
.filters label { font-size: 13px; color: #555; }
.filters input { padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; }
.filters button { padding: 4px 12px; cursor: pointer; }
table { border-collapse: collapse; width: 100%; margin-bottom: 16px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #e5e7eb; }
th { background: #f3f4f6; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.muted { color: #6b7280; }
.empty { color: #6b7280; font-style: italic; padding: 8px 0; }
.stub { padding: 30px; text-align: center; color: #6b7280; }
a.run { color: #4f46e5; text-decoration: none; } a.run:hover { text-decoration: underline; }
section { margin-bottom: 24px; }
section h2 { font-size: 16px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }
.event { padding: 4px 10px; border-left: 3px solid #6366f1; margin-bottom: 4px;
         background: #fff; } .event.tool { border-color: #10b981; }
.event.err { border-color: #ef4444; }
.tag { font-size: 11px; padding: 1px 6px; border-radius: 8px; background: #e5e7eb;
       color: #374151; }
"""


def _nav(active: str) -> str:
    items = [("Сводка", "/"), ("Прогоны", "/runs"), ("Аудит", "/audit")]
    links = " ".join(
        f'<a href="{href}"{_active_attr(name, active)}>{name}</a>'
        for name, href in items
    )
    return f"<header><strong>dev-helper-bot observability</strong> {links}</header>"


def _active_attr(name: str, active: str) -> str:
    return ' class="active"' if name == active else ""


def _filters_block() -> str:
    return """
<div class="filters" id="filters">
  <label>Метка <input type="text" name="label" id="f-label"></label>
  <label>С даты <input type="text" name="since" id="f-since"
                       placeholder="2026-09-01"></label>
  <button onclick="applyFilters()">Применить</button>
</div>
"""


def _poll_script(api_path: str, render_fn: str) -> str:
    """JS: опрашивает /api/{api_path} раз в POLL_INTERVAL_MS, перерисовывает
    блок #content функцией render{render_fn}(data). run_id в URL не сбрасывается."""
    return f"""
<script>
const POLL_MS = {POLL_INTERVAL_MS};
function row(k, v) {{ return '<tr><td>' + k + '</td><td>' + v + '</td></tr>'; }}
function currentQuery() {{
  const p = new URLSearchParams(location.search);
  const q = [];
  if (p.get('label')) q.push('label=' + encodeURIComponent(p.get('label')));
  if (p.get('since')) q.push('since=' + encodeURIComponent(p.get('since')));
  return q.length ? '?' + q.join('&') : '';
}}
function applyFilters() {{
  const label = document.getElementById('f-label').value;
  const since = document.getElementById('f-since').value;
  const p = new URLSearchParams();
  if (label) p.set('label', label);
  if (since) p.set('since', since);
  location.search = p.toString();
}}
function fillFilters() {{
  const p = new URLSearchParams(location.search);
  document.getElementById('f-label').value = p.get('label') || '';
  document.getElementById('f-since').value = p.get('since') || '';
}}
async function poll() {{
  try {{
    const r = await fetch('/api/{api_path}' + currentQuery());
    const data = await r.json();
    document.getElementById('content').innerHTML = {render_fn}(data);
  }} catch (e) {{
    document.getElementById('content').innerHTML =
      '<p class="empty">Ошибка опроса: ' + e.message + '</p>';
  }}
}}
fillFilters();
poll();
setInterval(poll, POLL_MS);
</script>
"""


def stub_page() -> web.Response:
    """Заглушка «телеметрия недоступна» (design D2, task 2.3)."""
    body = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>Телеметрия недоступна</title>
<style>{_CSS}</style></head>
<body>{_nav("")}<main>
<div class="stub">
  <h2>Телеметрия недоступна</h2>
  <p>Хранилище телеметрии не открылось при старте бота (сбой БД).<br>
  Бот продолжает работу; наблюдение в этом процессе недоступно.<br>
  CLI-скрипты (<code>obs-dashboard.py</code>, <code>obs-audit.py</code>)
  читают БД напрямую и могут работать, если файл доступен.</p>
</div>
</main></body></html>"""
    return web.Response(text=body, content_type="text/html")


_SUMMARY_JS = """
function renderSummary(d) {
  if (!d.available) return '<div class="stub"><h2>Телеметрия недоступна</h2></div>';
  const s = d.summary;
  if (!s.total_runs) return '<p class="empty">В базе телеметрии нет прогонов.</p>';
  let h = '<section><h2>Агрегаты</h2><table>';
  h += row('Прогонов', s.total_runs + ' (завершено ' + s.finished_runs + ')');
  h += row('Входные токены', s.input_tokens);
  h += row('Выходные токены', s.output_tokens);
  h += row('Кэшированные', s.cached_tokens == null
    ? '<span class="muted">недоступно (поставщик не отдаёт)</span>' : s.cached_tokens);
  h += row('Reasoning', !s.reasoning_available
    ? '<span class="muted">недоступно (поставщик не отдаёт)</span>'
    : (s.reasoning_tokens == null ? '0' : s.reasoning_tokens));
  h += row('Стоимость (виртуальная)', '$' + (s.estimated_cost || 0).toFixed(6));
  h += row('LLM-ходов', s.llm_calls);
  h += row('Tool-вызовов', s.tool_calls);
  if (s.cache_hit_rate != null)
    h += row('Cache hit rate', s.cache_hit_rate.toFixed(0) + '%');
  h += '</table></section>';
  if (s.status_breakdown.length) {
    h += '<section><h2>По статусам</h2><table>';
    h += '<tr><th>Статус</th><th class="num">Кол-во</th></tr>';
    for (const [st, c] of s.status_breakdown)
      h += '<tr><td>' + (st || 'не завершён') + '</td><td class="num">' + c + '</td></tr>';
    h += '</table></section>';
  }
  if (s.top_tools.length) {
    h += '<section><h2>Топ инструментов</h2><table>';
    h += '<tr><th>Инструмент</th><th class="num">Токены</th>'
      + '<th class="num">Доля</th><th>Вызовов</th></tr>';
    for (const t of s.top_tools)
      h += '<tr><td>' + t.name + '</td><td class="num">' + t.output_tokens
        + '</td><td class="num">' + t.share_percent.toFixed(1)
        + '%</td><td class="num">' + t.calls + '</td></tr>';
    h += '</table></section>';
  }
  return h;
}
"""


async def handle_summary_page(request: web.Request) -> web.Response:
    if request.app.get(STORE_KEY) is None:
        return stub_page()
    body = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>Сводка телеметрии</title>
<style>{_CSS}</style></head>
<body>{_nav("Сводка")}<main>
{_filters_block()}
<div id="content"><p class="empty">Загрузка…</p></div>
<script>
{_SUMMARY_JS}
</script>
{_poll_script("summary", "renderSummary")}
</main></body></html>"""
    return web.Response(text=body, content_type="text/html")


_RUNS_JS = """
function renderRuns(d) {
  if (!d.available) return '<div class="stub"><h2>Телеметрия недоступна</h2></div>';
  if (!d.runs.length) return '<p class="empty">В базе телеметрии нет прогонов.</p>';
  let h = '<table><tr><th>ID</th><th>Чат</th><th>Метка</th><th>Начало</th>'
    + '<th>Статус</th><th class="num">LLM</th><th class="num">Tool</th>'
    + '<th class="num">Вход</th><th class="num">Выход</th>'
    + '<th class="num">Стоимость</th></tr>';
  for (const r of d.runs) {
    h += '<tr><td><a class="run" href="/runs/' + r.id + '">' + r.id + '</a></td>'
      + '<td>' + r.chat_id + '</td>'
      + '<td>' + (r.label || '<span class="muted">—</span>') + '</td>'
      + '<td>' + r.started_at + '</td>'
      + '<td>' + (r.status || '<span class="muted">не завершён</span>') + '</td>'
      + '<td class="num">' + (r.llm_calls == null ? '—' : r.llm_calls) + '</td>'
      + '<td class="num">' + (r.tool_calls == null ? '—' : r.tool_calls) + '</td>'
      + '<td class="num">' + (r.input_tokens == null ? '—' : r.input_tokens) + '</td>'
      + '<td class="num">' + (r.output_tokens == null ? '—' : r.output_tokens) + '</td>'
      + '<td class="num">' + (r.estimated_cost == null ? '—'
        : '$' + r.estimated_cost.toFixed(6)) + '</td></tr>';
  }
  h += '</table>';
  return h;
}
"""


async def handle_runs_page(request: web.Request) -> web.Response:
    if request.app.get(STORE_KEY) is None:
        return stub_page()
    body = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>Прогоны</title>
<style>{_CSS}</style></head>
<body>{_nav("Прогоны")}<main>
{_filters_block()}
<div id="content"><p class="empty">Загрузка…</p></div>
<script>
{_RUNS_JS}
</script>
{_poll_script("runs", "renderRuns")}
</main></body></html>"""
    return web.Response(text=body, content_type="text/html")


_RUN_JS = """
function renderRun(d) {
  if (!d.available) return '<div class="stub"><h2>Телеметрия недоступна</h2></div>';
  if (!d.run) return '<p class="empty">Прогон не найден.</p>';
  const r = d.run.run;
  let h = '<section><h2>Прогон #' + r.id + '</h2><table>';
  h += row('Чат', r.chat_id);
  h += row('Метка', r.label || '<span class="muted">—</span>');
  h += row('Начало', r.started_at);
  h += row('Завершение', r.finished_at || '…');
  h += row('Статус', r.status || '<span class="muted">не завершён</span>');
  h += row('LLM-ходов', r.llm_calls == null ? '—' : r.llm_calls);
  h += row('Tool-вызовов', r.tool_calls == null ? '—' : r.tool_calls);
  h += row('Входные', r.input_tokens == null ? '—' : r.input_tokens);
  h += row('Выходные', r.output_tokens == null ? '—' : r.output_tokens);
  h += row('Кэшированные', r.cached_tokens == null
    ? '<span class="muted">недоступно</span>' : r.cached_tokens);
  h += row('Стоимость', r.estimated_cost == null ? '—'
    : '$' + r.estimated_cost.toFixed(6));
  h += '</table></section><section><h2>Timeline</h2>';
  const llm = d.run.llm_calls || [], tools = d.run.tool_calls || [];
  if (!llm.length && !tools.length) h += '<p class="empty">Событий нет.</p>';
  const events = [];
  for (const c of llm) events.push([c.turn_number, 0, c.id, c]);
  for (const c of tools) events.push([c.turn_number, 1, c.id, c]);
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  for (const [, , , e] of events) h += renderEvent(e);
  h += '</section>';
  return h;
}
function renderEvent(e) {
  if (e.tool_name !== undefined) {
    return '<div class="event tool">Turn ' + e.turn_number + ' &middot; '
      + '<span class="tag">tool</span> <strong>' + e.tool_name + '</strong> '
      + (e.ok ? 'успех' : 'ошибка') + ', ' + Math.round(e.duration_ms) + 'мс, '
      + '~' + (e.output_tokens == null ? '?' : e.output_tokens) + ' ток. '
      + '(арг. ' + e.input_size + ', рез. ' + e.output_size + ')</div>';
  }
  if (!e.ok) {
    return '<div class="event err">Turn ' + e.turn_number + ' &middot; '
      + '<span class="tag">LLM</span> ошибка недоступности, '
      + Math.round(e.latency_ms) + 'мс</div>';
  }
  let s = '<div class="event">Turn ' + e.turn_number + ' &middot; '
    + '<span class="tag">LLM</span> ';
  s += 'вх ' + (e.input_tokens == null ? '?' : e.input_tokens)
    + ' вых ' + (e.output_tokens == null ? '?' : e.output_tokens);
  if (e.cached_tokens != null) s += ' кэш ' + e.cached_tokens;
  if (e.reasoning_tokens != null) s += ' reasoning ' + e.reasoning_tokens;
  s += ' ' + Math.round(e.latency_ms) + 'мс';
  if (e.estimated_cost != null) s += ' $' + e.estimated_cost.toFixed(6);
  s += '</div>';
  return s;
}
"""


async def handle_run_page(request: web.Request) -> web.Response:
    if request.app.get(STORE_KEY) is None:
        return stub_page()
    run_id = request.match_info["run_id"]
    body = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>Прогон #{run_id}</title>
<style>{_CSS}</style></head>
<body>{_nav("Прогоны")}<main>
<p><a href="/runs">&larr; К списку прогонов</a></p>
<div id="content"><p class="empty">Загрузка…</p></div>
<script>
{_RUN_JS}
</script>
{_poll_script("runs/" + run_id, "renderRun")}
</main></body></html>"""
    return web.Response(text=body, content_type="text/html")


_AUDIT_JS = """
function renderAudit(d) {
  if (!d.available) return '<div class="stub"><h2>Телеметрия недоступна</h2></div>';
  const a = d.audit;
  if (!a.runs_count) return '<p class="empty">В базе телеметрии нет прогонов.</p>';
  let h = '<section><h2>Аудит: прогонов ' + a.runs_count + '</h2></section>';
  h += renderQ1(a);
  h += renderQ2(a);
  h += renderQ3(a);
  h += renderQ4(a);
  h += renderQ5(a);
  return h;
}
function renderQ1(a) {
  let h = '<section><h2>Q1. Топ инструментов (оценка вывода, chars/4)</h2>';
  if (!a.top_tools.length) { return h + '<p class="empty">Нет вызовов инструментов.</p></section>'; }
  h += '<table><tr><th>Инструмент</th><th class="num">Токены</th>'
    + '<th class="num">Доля</th><th class="num">Вызовов</th></tr>';
  for (const t of a.top_tools)
    h += '<tr><td>' + t.name + '</td><td class="num">' + t.output_tokens
      + '</td><td class="num">' + t.share_percent.toFixed(1)
      + '%</td><td class="num">' + t.calls + '</td></tr>';
  return h + '</table></section>';
}
function renderQ2(a) {
  let h = '<section><h2>Q2. Самый дорогой ход по входному контексту</h2>';
  if (!a.expensive_turns.length)
    return h + '<p class="empty">Нет LLM-вызовов с входными токенами.</p></section>';
  if (a.single_turn_only) {
    const t = a.expensive_turns[0];
    h += '<p>Ход ' + t.turn + ': среднее ' + Math.round(t.avg_input_tokens)
      + ' input токенов (' + t.runs + ' прогон.). Все прогоны по одному '
      + 'LLM-ходу — динамики роста нет.</p>';
    return h + '</section>';
  }
  h += '<table><tr><th>Ход</th><th class="num">Среднее input</th>'
    + '<th class="num">Прогонов</th><th>Рост</th></tr>';
  for (const t of a.expensive_turns)
    h += '<tr><td>' + t.turn + '</td><td class="num">'
      + Math.round(t.avg_input_tokens) + '</td><td class="num">' + t.runs
      + '</td><td>' + (t.growth_percent == null ? '—'
      : (t.growth_percent >= 0 ? '+' : '') + t.growth_percent.toFixed(0) + '%')
      + '</td></tr>';
  h += '</table><p>Самый дорогой ход: <strong>' + a.most_expensive_turn
    + '</strong>.</p></section>';
  return h;
}
function renderQ3(a) {
  let h = '<section><h2>Q3. Рост типов контекста (по разбивке prompt_roles)</h2>';
  if (!a.context_types)
    return h + '<p class="empty">Недоступно: записи без разбивки по ролям '
      + '(prompt_roles null).</p></section>';
  const ct = a.context_types;
  if (ct.skipped_no_roles)
    h += '<p class="muted">(без разбивки, не учтено: ' + ct.skipped_no_roles
      + ' вызов.)</p>';
  h += '<p>Всего символов контекста: ' + ct.total_chars + '</p>';
  h += '<table><tr><th>Тип</th><th class="num">Символы</th>'
    + '<th class="num">Доля</th></tr>';
  for (const name in ct.shares)
    h += '<tr><td>' + name + '</td><td class="num">' + (ct.totals[name] || 0)
      + '</td><td class="num">' + (ct.shares[name] == null ? '?' : ct.shares[name].toFixed(1))
      + '%</td></tr>';
  h += '</table>';
  if (Object.keys(ct.trend_by_turn).length) {
    h += '<p>Тренд долей по ходам (среднее по прогонам):</p>';
    for (const turn in ct.trend_by_turn) {
      const shares = ct.trend_by_turn[turn];
      const parts = Object.keys(shares).map(k =>
        k + ' ' + Math.round(shares[k]) + '%').join('  ');
      h += '<p>ход ' + turn + ': ' + parts + '</p>';
    }
  }
  return h + '</section>';
}
function renderQ4(a) {
  let h = '<section><h2>Q4. Повторно отправляемые токены</h2>';
  if (!a.repeats)
    return h + '<p class="empty">Нет прогонов с входными токенами.</p></section>';
  const r = a.repeats;
  h += '<table>';
  h += row('Всего входных', r.total_input);
  h += row('Новая информация', r.new_tokens);
  h += row('Повторная информация', r.repeated_tokens + '  ('
    + (r.repeated_share_percent == null ? '?' : r.repeated_share_percent.toFixed(0))
    + '% — модель уже видела)');
  h += '</table><p>Межпрогонный слой (оценка сверху): ~' + r.inter_run_tokens
    + ' токенов истории сессии, ре-отправляемой из прогона в прогон.</p></section>';
  return h;
}
function renderQ5(a) {
  let h = '<section><h2>Q5. Стоимость: фактическая и оценки по прайсу</h2>';
  if (!a.cost)
    return h + '<p class="empty">Нет LLM-вызовов с входными токенами.</p></section>';
  const c = a.cost;
  h += '<table>';
  if (c.billed_total != null)
    h += row('Фактический биллинг поставщика', c.billed_total.toFixed(4)
      + ' ₽ (по ' + c.billed_calls + ' вызовам)');
  if (c.raw_cost != null) {
    const lbl = c.billed_total != null ? 'Сырая оценка' : 'Сырая стоимость';
    h += row(lbl, '$' + c.raw_cost.toFixed(6));
  }
  if (c.effective_cost != null) {
    const lbl = c.billed_total != null ? 'Эффективная оценка' : 'Эффективная стоимость';
    h += row(lbl, '$' + c.effective_cost.toFixed(6)
      + (c.has_cache_data ? ' (кэш ' + (c.cached_share_percent == null ? 0
      : c.cached_share_percent).toFixed(1) + '%)' : ' — данных о кэше нет'));
  }
  if (c.cache_savings != null)
    h += row('Экономия кэша', '$' + c.cache_savings.toFixed(6));
  h += '</table></section>';
  return h;
}
"""


async def handle_audit_page(request: web.Request) -> web.Response:
    if request.app.get(STORE_KEY) is None:
        return stub_page()
    body = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>Аудит токенов</title>
<style>{_CSS}</style></head>
<body>{_nav("Аудит")}<main>
{_filters_block()}
<div id="content"><p class="empty">Загрузка…</p></div>
<script>
{_AUDIT_JS}
</script>
{_poll_script("audit", "renderAudit")}
</main></body></html>"""
    return web.Response(text=body, content_type="text/html")
