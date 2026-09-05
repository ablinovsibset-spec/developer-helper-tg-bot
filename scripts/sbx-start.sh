#!/usr/bin/env bash
# Ежедневный запуск бота внутри сандбокса Docker Sandboxes (design D6/D7).
#
# Что делает:
#   1. поднимает остановленный сандбокс (sbx exec сам стартует его — спайк 1.2);
#   2. останавливает предыдущий процесс бота, если он жив (сценарий
#      обновления кода: правки на хосте + повторный запуск этого скрипта);
#   3. запускает бот в фоне с явным override LLM_BASE_URL (побеждает
#      localhost из workspace-.env: dotenv не перезаписывает существующее
#      окружение), логи — в /tmp/bot.log внутри VM;
#   4. открывает keepalive-сессию: демон Docker Sandboxes останавливает
#      сандбокс через ~30с после отключения последней exec-сессии, а фоновый
#      процесс бота сессией не считается — без удержания бот умирает;
#   5. показывает хвост лога.
#
# Токен Telegram приходит из workspace-.env через load_dotenv() внутри VM;
# командам агента .env недоступен (граница доверия — контейнер-жилец).
#
# Остановка/пауза: scripts/sbx-stop.sh. После ребута хоста — запустить
# этот скрипт заново (автозапуска нет, восстановление ручное).
set -euo pipefail

SANDBOX_NAME="${SBX_NAME:-devbot}"
LLM_PORT="${LLM_PORT:-1234}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_BASE_URL="${LLM_BASE_URL:-http://host.docker.internal:${LLM_PORT}/v1}"
KEEPALIVE_PID_FILE="/tmp/dev-helper-bot-keepalive-${SANDBOX_NAME}.pid"

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
sbx exec "${SANDBOX_NAME}" bash -c 'test -x "$HOME/.venv-devbot/bin/python"' \
    || die "venv ~/.venv-devbot не найден внутри сандбокса. Сначала: scripts/sbx-setup.sh"

say "Останавливаю предыдущий процесс бота (если был)"
# Паттерн ловит обе формы запуска: модульную (dev_helper_bot.main) и
# консольный скрипт (dev-helper-bot). Квадратные скобки — чтобы pkill -f
# не убил собственный wrapper, чья команда содержит этот же текст.
sbx exec "${SANDBOX_NAME}" bash -c "pkill -f '[d]ev.helper.bot' || true"

sbx exec "${SANDBOX_NAME}" bash -c "echo \"=== bot start \$(date '+%Y-%m-%d %H:%M:%S') ===\" >> /tmp/bot.log"

say "Запускаю бот в фоне с LLM_BASE_URL=${LLM_BASE_URL} (логи: /tmp/bot.log внутри VM)"
# Отделяемся классически (setsid + перенаправление всех потоков внутри VM):
# процесс переживает закрытие exec-сессии и хостового клиента.
sbx exec -e "LLM_BASE_URL=${LLM_BASE_URL}" "${SANDBOX_NAME}" \
    bash -c "cd '${WORKSPACE}' && setsid nohup \$HOME/.venv-devbot/bin/python -m dev_helper_bot.main >> /tmp/bot.log 2>&1 < /dev/null & echo 'бот запущен (pid '\$!')'"

say "Открываю keepalive-сессию (удерживает сандбокс от автостопа)"
# Убиваем предыдущий keepalive-клиент, если жив, и открываем новый.
if [ -f "${KEEPALIVE_PID_FILE}" ]; then
    kill "$(cat "${KEEPALIVE_PID_FILE}")" >/dev/null 2>&1 || true
    rm -f "${KEEPALIVE_PID_FILE}"
fi
nohup sbx exec "${SANDBOX_NAME}" sleep infinity >/dev/null 2>&1 &
echo $! > "${KEEPALIVE_PID_FILE}"

say "Жду 3с и показываю хвост лога"
sleep 3
sbx exec "${SANDBOX_NAME}" tail -n 50 /tmp/bot.log
say "Готово. Пауза/остановка: scripts/sbx-stop.sh; keepalive pid: $(cat "${KEEPALIVE_PID_FILE}")"
