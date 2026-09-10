#!/usr/bin/env bash
# Ежедневный запуск бота внутри сандбокса Docker Sandboxes (design D6/D7).
#
# Что делает:
#   1. поднимает остановленный сандбокс (sbx exec сам стартует его — спайк 1.2);
#   2. интерактивно запрашивает LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
#      (дефолты — LM Studio + openai/gpt-oss-20b + пустой ключ; Enter принимает);
#   3. останавливает предыдущий процесс бота, если он жив (сценарий
#      обновления кода: правки на хосте + повторный запуск этого скрипта);
#   4. запускает бот в фоне с явным override всех трёх LLM_* (побеждают
#      значения из workspace-.env: dotenv не перезаписывает существующее
#      окружение), логи — в /tmp/bot.log внутри VM;
#   5. открывает keepalive-сессию: демон Docker Sandboxes останавливает
#      сандбокс через ~30с после отключения последней exec-сессии, а фоновый
#      процесс бота сессией не считается — без удержания бот умирает;
#   6. показывает хвост лога.
#
# Токен Telegram приходит из workspace-.env через load_dotenv() внутри VM;
# командам агента .env недоступен (граница доверия — контейнер-жилец).
# Скрипт .env не переписывает — выбранные LLM_* действуют только на этот запуск.
#
# Остановка/пауза: scripts/sbx-stop.sh. После ребута хоста — запустить
# этот скрипт заново (автозапуска нет, восстановление ручное).
set -euo pipefail

SANDBOX_NAME="${SBX_NAME:-devbot}"
LLM_PORT="${LLM_PORT:-1234}"
OBS_WEB_PORT="${OBS_WEB_PORT:-8765}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEEPALIVE_PID_FILE="/tmp/dev-helper-bot-keepalive-${SANDBOX_NAME}.pid"

# Pre-fill: уже экспортированные LLM_* из вызывающего shell; иначе — LM Studio.
DEFAULT_LLM_BASE_URL="${LLM_BASE_URL:-http://host.docker.internal:${LLM_PORT}/v1}"
DEFAULT_LLM_MODEL="${LLM_MODEL:-openai/gpt-oss-20b}"
DEFAULT_LLM_API_KEY="${LLM_API_KEY:-}"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mОшибка:\033[0m %s\n' "$*" >&2; exit 1; }

command -v sbx >/dev/null || die "sbx CLI не найден. Установка: brew install --cask sbx"

# Демон sandboxd не поднимается командами сам по себе — стартуем при необходимости.
if ! sbx ls >/dev/null 2>&1; then
    sbx daemon start -d >/dev/null 2>&1 || true
    sleep 1
    sbx ls >/dev/null 2>&1 || die "sbx не авторизован или демон недоступен: sbx login / sbx daemon start -d"
fi

sbx ls 2>/dev/null | awk '{print $1}' | grep -qx "${SANDBOX_NAME}" \
    || die "сандбокс '${SANDBOX_NAME}' не найден. Сначала: scripts/sbx-setup.sh"

# Первый exec заодно поднимает остановленный сандбокс (спайк 1.2).
# </dev/null — чтобы sbx exec не съел stdin до интерактивных промптов LLM.
sbx exec "${SANDBOX_NAME}" bash -c 'test -x "$HOME/.venv-devbot/bin/python"' </dev/null \
    || die "venv ~/.venv-devbot не найден внутри сандбокса. Сначала: scripts/sbx-setup.sh"

say "Параметры LLM для этого запуска (Enter — принять значение в [...])"
printf 'LLM_BASE_URL [%s]: ' "${DEFAULT_LLM_BASE_URL}"
read -r LLM_BASE_URL_INPUT
LLM_BASE_URL="${LLM_BASE_URL_INPUT:-${DEFAULT_LLM_BASE_URL}}"

printf 'LLM_MODEL [%s]: ' "${DEFAULT_LLM_MODEL}"
read -r LLM_MODEL_INPUT
LLM_MODEL="${LLM_MODEL_INPUT:-${DEFAULT_LLM_MODEL}}"

if [ -n "${DEFAULT_LLM_API_KEY}" ]; then
    printf 'LLM_API_KEY [set — Enter сохранить, или введите новый]: '
else
    printf 'LLM_API_KEY [empty]: '
fi
read -rs LLM_API_KEY_INPUT
printf '\n'
LLM_API_KEY="${LLM_API_KEY_INPUT:-${DEFAULT_LLM_API_KEY}}"

if [ -n "${LLM_API_KEY}" ]; then
    LLM_API_KEY_STATUS=set
else
    LLM_API_KEY_STATUS=empty
fi
say "LLM: URL=${LLM_BASE_URL} model=${LLM_MODEL} key=${LLM_API_KEY_STATUS}"

say "Останавливаю предыдущий процесс бота (если был)"
# Паттерн ловит обе формы запуска: модульную (dev_helper_bot.main) и
# консольный скрипт (dev-helper-bot). Квадратные скобки — чтобы pkill -f
# не убил собственный wrapper, чья команда содержит этот же текст.
sbx exec "${SANDBOX_NAME}" bash -c "pkill -f '[d]ev.helper.bot' || true" </dev/null

sbx exec "${SANDBOX_NAME}" bash -c "echo \"=== bot start \$(date '+%Y-%m-%d %H:%M:%S') ===\" >> /tmp/bot.log" </dev/null

say "Запускаю бот в фоне (логи: /tmp/bot.log внутри VM)"
# Отделяемся классически (setsid + перенаправление всех потоков внутри VM):
# процесс переживает закрытие exec-сессии и хостового клиента.
# Все три LLM_* явно в -e, включая пустой ключ — иначе облачный ключ из
# workspace-.env «протечёт» в локальный LM Studio-сеанс после Enter×3.
# OBS_WEB_HOST=0.0.0.0: дашборд биндится на все интерфейсы VM, чтобы проброс
# sbx ports (вход через сетевой интерфейс VM) доставал до него; на хост
# публикуется только loopback (см. config.py / design D6).
sbx exec \
    -e "LLM_BASE_URL=${LLM_BASE_URL}" \
    -e "LLM_MODEL=${LLM_MODEL}" \
    -e "LLM_API_KEY=${LLM_API_KEY}" \
    -e "OBS_WEB_HOST=0.0.0.0" \
    "${SANDBOX_NAME}" \
    bash -c "cd '${WORKSPACE}' && setsid nohup \$HOME/.venv-devbot/bin/python -m dev_helper_bot.main >> /tmp/bot.log 2>&1 < /dev/null & echo 'бот запущен (pid '\$!')'" \
    </dev/null

say "Открываю keepalive-сессию (удерживает сандбокс от автостопа)"
# Убиваем предыдущий keepalive-клиент, если жив, и открываем новый.
if [ -f "${KEEPALIVE_PID_FILE}" ]; then
    kill "$(cat "${KEEPALIVE_PID_FILE}")" >/dev/null 2>&1 || true
    rm -f "${KEEPALIVE_PID_FILE}"
fi
nohup sbx exec "${SANDBOX_NAME}" sleep infinity </dev/null >/dev/null 2>&1 &
echo $! > "${KEEPALIVE_PID_FILE}"

# Публикуем порт веб-дашборда телеметрии на хост (change add-obs-web-dashboard):
# дашборд биндится на 127.0.0.1:OBS_WEB_PORT *внутри* VM, а браузер на хосте
# до loopback VM не достаёт. sbx ports пробрасывает VM:OBS_WEB_PORT на
# 127.0.0.1:OBS_WEB_PORT хоста. Идемпотентно: повторный запуск не дублирует.
say "Публикую порт дашборда ${OBS_WEB_PORT} (http://127.0.0.1:${OBS_WEB_PORT}/)"
sbx ports "${SANDBOX_NAME}" --publish "${OBS_WEB_PORT}:${OBS_WEB_PORT}" >/dev/null 2>&1 || \
    say "предупреждение: не удалось опубликовать порт ${OBS_WEB_PORT} (возможно, уже опубликован)"

say "Жду 3с и показываю хвост лога"
sleep 3
sbx exec "${SANDBOX_NAME}" tail -n 50 /tmp/bot.log </dev/null
say "Готово. Дашборд: http://127.0.0.1:${OBS_WEB_PORT}/  Пауза/остановка: scripts/sbx-stop.sh; keepalive pid: $(cat "${KEEPALIVE_PID_FILE}")"
