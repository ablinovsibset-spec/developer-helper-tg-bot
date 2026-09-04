#!/usr/bin/env bash
# Одноразовая подготовка сандбокса Docker Sandboxes для бота (design D4/D7).
#
# Что делает:
#   1. проверяет sbx CLI и авторизацию (подсказка: sbx login);
#   2. добавляет разовую network policy для доступа к LLM на хосте;
#   3. создаёт сандбокс `devbot` (шаблон shell — «голая» agent-less VM,
#      спайк 1.1) с workspace-маунтом этого репозитория;
#   4. ставит python-venv на VM-локальном диске (~/.venv-devbot, вне
#      workspace) и выполняет editable-установку пакета бота.
#
# Повторный запуск безопасен: существующий сандбокс переиспользуется,
# venv и пакет обновляются.
set -euo pipefail

SANDBOX_NAME="${SBX_NAME:-devbot}"
LLM_PORT="${LLM_PORT:-1234}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mОшибка:\033[0m %s\n' "$*" >&2; exit 1; }

# sbx CLI (0.39.0) спорадически зависает на старте (дедлок после запроса
# feature-flags) — каждый вызов sbx идёт через timeout. GNU timeout на
# macOS нет, поэтому fallback через perl alarm (сигнал переживает exec).
SBX_FAST_TIMEOUT="${SBX_FAST_TIMEOUT:-30}"    # ls/policy
SBX_SLOW_TIMEOUT="${SBX_SLOW_TIMEOUT:-900}"   # create/exec (apt, pip)
sbx_to() {
    local secs="$1"; shift
    if command -v timeout >/dev/null 2>&1; then
        timeout "$secs" "$@"
    elif command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$secs" "$@"
    else
        perl -e 'alarm shift; exec @ARGV' "$secs" "$@"
    fi
}

command -v sbx >/dev/null || die "sbx CLI не найден. Установка: brew install --cask sbx (см. docs.docker.com/ai/sandboxes/install)"

# Авторизация: любая ошибка списка сандбоксов про login — подсказка.
if ! sbx_to "${SBX_FAST_TIMEOUT}" sbx ls >/dev/null 2>&1; then
    die "sbx не авторизован, демон недоступен или CLI завис (timeout ${SBX_FAST_TIMEOUT}s). Выполните: sbx login (запуск демона: sbx daemon start -d)"
fi

# Разовая network policy: LLM-порт хоста достижим как host.docker.internal:<port>.
if ! sbx_to "${SBX_FAST_TIMEOUT}" sbx policy ls 2>/dev/null | grep -q "localhost:${LLM_PORT}"; then
    say "Добавляю network policy localhost:${LLM_PORT} (разовая операция)"
    sbx_to "${SBX_FAST_TIMEOUT}" sbx policy allow network "localhost:${LLM_PORT}"
fi

# Создание сандбокса с workspace (шаблон shell, спайк 1.1). Workspace
# смотрирован по абсолютному пути хоста — он совпадает внутри VM.
if sbx_to "${SBX_FAST_TIMEOUT}" sbx ls 2>/dev/null | awk '{print $1}' | grep -qx "${SANDBOX_NAME}"; then
    say "Сандбокс '${SANDBOX_NAME}' уже существует — переиспользую"
else
    say "Создаю сандбокс '${SANDBOX_NAME}' (шаблон shell) с workspace ${WORKSPACE}"
    sbx_to "${SBX_SLOW_TIMEOUT}" sbx create --name "${SANDBOX_NAME}" shell "${WORKSPACE}"
fi

# Пакет venv отсутствует в базовом образе shell (Ubuntu без python3-venv);
# установка идемпотентна и переживает stop/start сандбокса.
say "Проверяю/ставлю python3-venv внутри сандбокса (apt, от root)"
sbx_to "${SBX_SLOW_TIMEOUT}" sbx exec -u root "${SANDBOX_NAME}" bash -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq python3-venv
'

# venv на VM-локальном диске (вне workspace — каталог окружения не попадает
# в репозиторий) + editable-установка: правки кода и скиллов на хосте
# подхватываются рестартом процесса бота без переустановки.
say "Готовлю python-окружение внутри сандбокса (~/.venv-devbot, editable-установка)"
sbx_to "${SBX_SLOW_TIMEOUT}" sbx exec "${SANDBOX_NAME}" bash -c "
    set -e
    python3 -m venv ~/.venv-devbot
    ~/.venv-devbot/bin/pip install --quiet --upgrade pip
    ~/.venv-devbot/bin/pip install --quiet -e '${WORKSPACE}'
    ~/.venv-devbot/bin/python -c 'import aiogram, dev_helper_bot; print(\"deps ok\")'
"

say "Готово. Запуск бота: scripts/sbx-start.sh (логи: sbx exec ${SANDBOX_NAME} tail -n 50 /tmp/bot.log)"
