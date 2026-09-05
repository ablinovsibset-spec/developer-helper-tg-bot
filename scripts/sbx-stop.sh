#!/usr/bin/env bash
# Пауза/остановка бота и сандбокса (ручная модель восстановления).
#
# Что делает:
#   1. закрывает keepalive-сессию sbx-start.sh (без неё демон автостопит
#      сандбокс через ~30с — можно было бы просто подождать);
#   2. останавливает процесс бота внутри VM;
#   3. останавливает сандбокс (состояние — пакеты, образы, диски —
#      сохраняется до sbx rm).
#
# Продолжение работы: scripts/sbx-start.sh.
set -euo pipefail

SANDBOX_NAME="${SBX_NAME:-devbot}"
KEEPALIVE_PID_FILE="/tmp/dev-helper-bot-keepalive-${SANDBOX_NAME}.pid"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mОшибка:\033[0m %s\n' "$*" >&2; exit 1; }

command -v sbx >/dev/null || die "sbx CLI не найден. Установка: brew install --cask sbx"

if [ -f "${KEEPALIVE_PID_FILE}" ]; then
    say "Закрываю keepalive-сессию"
    kill "$(cat "${KEEPALIVE_PID_FILE}")" >/dev/null 2>&1 || true
    rm -f "${KEEPALIVE_PID_FILE}"
fi

say "Останавливаю процесс бота и сандбокс '${SANDBOX_NAME}'"
sbx exec "${SANDBOX_NAME}" bash -c "pkill -f '[d]ev.helper.bot' || true" || true
sbx stop "${SANDBOX_NAME}" || true

say "Готово. Продолжение: scripts/sbx-start.sh; полный сброс: sbx rm ${SANDBOX_NAME} && scripts/sbx-setup.sh"
