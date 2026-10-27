"""Лог запросов к боту для аналитики покрытия базы знаний: logs/queries.jsonl, одна строка JSON на запрос."""
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "queries.jsonl"

logger = logging.getLogger("query_log")

# Бот отвечает на вопросы в нескольких потоках — без блокировки строки разных запросов могут перемешаться
_lock = threading.Lock()


def append(record):
    """Дописывает запрос в лог, добавляя время. Сбой записи не должен ломать ответ — он только попадает в лог бота."""
    line = json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"), **record}, ensure_ascii=False)
    try:
        LOG_FILE.parent.mkdir(exist_ok=True)
        with _lock, LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    except OSError:
        logger.exception("Не удалось записать запрос в %s", LOG_FILE)
