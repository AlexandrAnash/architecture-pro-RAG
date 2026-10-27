#!/bin/bash
# Ежедневное обновление индекса с повтором при сбое.
# Cron (каждый день в 6:00): 0 6 * * * /полный/путь/architecture-pro-RAG/scripts/update.sh
# Лог: logs/update_index.log — общий с update_index.py.
set -u

cd "$(dirname "$0")/.." || exit 1

LOG=logs/update_index.log
ATTEMPTS=3
RETRY_PAUSE_SECONDS=300
EXIT_FAILED=2  # код update_index.py, когда индекс не обновлен — только тогда есть смысл повторять

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1 update.sh: $2" >>"$LOG"
}

mkdir -p logs
for attempt in $(seq "$ATTEMPTS"); do
    log INFO "попытка $attempt из $ATTEMPTS"
    # строки лога update_index.py сам пишет в файл; с экрана берем только ошибки вне лога (например, сбой импорта)
    .venv/bin/python scripts/update_index.py >/dev/null 2>>"$LOG"
    code=$?
    if [ "$code" -ne "$EXIT_FAILED" ]; then
        exit "$code"
    fi
    if [ "$attempt" -lt "$ATTEMPTS" ]; then
        log WARNING "индекс не обновлен, повтор через $RETRY_PAUSE_SECONDS с"
        sleep "$RETRY_PAUSE_SECONDS"
    fi
done
log ERROR "индекс не обновлен после $ATTEMPTS попыток"
exit "$EXIT_FAILED"
