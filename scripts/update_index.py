"""Обновляет индекс базы знаний (knowledge_base/ -> chroma_db/) без полной пересборки.

По хешу содержимого находит новые, измененные и удаленные документы и меняет только их чанки.
Запуск вручную: .venv/bin/python scripts/update_index.py
Ежедневный запуск — через scripts/update.sh (повторяет попытку при сбое).
Лог: logs/update_index.log и экран.
Код выхода: 0 — все хорошо, 1 — часть документов пропущена из-за ошибок, 2 — индекс не обновлен.
"""
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import chromadb

from build_index import (
    COLLECTION,
    COLLECTION_METADATA,
    DOCS_DIR,
    INDEX_DIR,
    NO_TELEMETRY,
    ROOT,
    file_chunks,
    file_hash,
    list_documents,
    load_model,
    source_name,
)

LOG_FILE = ROOT / "logs" / "update_index.log"
BYTES_IN_MB = 1024 * 1024

EXIT_OK = 0
EXIT_DOCUMENTS_SKIPPED = 1
EXIT_FAILED = 2

logger = logging.getLogger("update_index")


@dataclass
class Report:
    files_added: int = 0
    files_changed: int = 0
    files_deleted: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    errors: int = 0

    def summary(self):
        return (
            f"index updated at {datetime.now():%Y-%m-%d %H:%M}, {self.files_added} files added, "
            f"{self.files_changed} changed, {self.files_deleted} deleted, {self.chunks_added} chunks added, "
            f"{self.chunks_deleted} chunks deleted, {self.errors} errors"
        )


def setup_logging():
    LOG_FILE.parent.mkdir(exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s update_index: %(message)s")
    for handler in (logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def open_collection():
    """Коллекция индекса; при первом запуске создается пустая — тогда в нее попадут все документы."""
    client = chromadb.PersistentClient(path=str(INDEX_DIR), settings=NO_TELEMETRY)
    return client.get_or_create_collection(COLLECTION, metadata=COLLECTION_METADATA)


def indexed_hashes(collection):
    """Документы, которые уже в индексе: {путь: хеш}. У индекса, собранного до появления хешей, хеш None."""
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    return {metadata["source"]: metadata.get("hash") for metadata in metadatas}


def find_changes(collection):
    """Сравнивает папку с индексом: новые и измененные документы (пути к файлам) и удаленные (имена в индексе)."""
    current = {source_name(path): path for path in list_documents(DOCS_DIR)}
    indexed = indexed_hashes(collection)
    added = [path for source, path in current.items() if source not in indexed]
    changed = [path for source, path in current.items() if source in indexed and file_hash(path) != indexed[source]]
    deleted = [source for source in indexed if source not in current]
    return added, changed, deleted


def log_changes(added, changed, deleted):
    logger.info("Найдено: новых %d, измененных %d, удаленных %d", len(added), len(changed), len(deleted))
    for path in added:
        logger.info("  + %s", source_name(path))
    for path in changed:
        logger.info("  ~ %s", source_name(path))
    for source in deleted:
        logger.info("  - %s", source)


def collect_chunks(paths):
    """Чанки документов. Документ, который не удалось прочитать, пропускается и попадает в список ошибок."""
    ids, texts, metadatas, failed = [], [], [], []
    for path in paths:
        try:
            file_ids, file_texts, file_metadatas = file_chunks(path)
        except (OSError, ValueError, IndexError) as error:
            logger.error("Документ %s пропущен: %s", source_name(path), error)
            failed.append(path)
            continue
        ids += file_ids
        texts += file_texts
        metadatas += file_metadatas
    return ids, texts, metadatas, failed


def delete_chunks(collection, sources):
    """Удаляет все чанки указанных документов. Возвращает, сколько удалено."""
    if not sources:
        return 0
    ids = collection.get(where={"source": {"$in": sources}}, include=[])["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def update(collection, model):
    added, changed, deleted = find_changes(collection)
    log_changes(added, changed, deleted)
    ids, texts, metadatas, failed = collect_chunks(added + changed)
    # Эмбеддинги считаем до изменений в индексе: если упадет здесь, индекс останется прежним
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist() if texts else []
    # У измененного документа, который не удалось прочитать, оставляем старые чанки: старая версия лучше, чем никакой
    changed_ok = [path for path in changed if path not in failed]
    chunks_deleted = delete_chunks(collection, [source_name(path) for path in changed_ok] + deleted)
    if ids:
        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    return Report(
        files_added=len([path for path in added if path not in failed]),
        files_changed=len(changed_ok),
        files_deleted=len(deleted),
        chunks_added=len(ids),
        chunks_deleted=chunks_deleted,
        errors=len(failed),
    )


def index_size_mb():
    return sum(path.stat().st_size for path in INDEX_DIR.rglob("*") if path.is_file()) / BYTES_IN_MB


def main():
    setup_logging()
    started = time.perf_counter()
    logger.info("Запуск: %s -> %s", DOCS_DIR.relative_to(ROOT), INDEX_DIR.relative_to(ROOT))
    try:
        collection = open_collection()
        report = update(collection, load_model())
    # SystemExit тоже ловим: load_model так сообщает, что модель не скачана, и причина должна попасть в лог
    except (Exception, SystemExit):
        logger.exception("Обновление прервано")
        logger.error("index not updated at %s, 1 errors", f"{datetime.now():%Y-%m-%d %H:%M}")
        return EXIT_FAILED
    logger.info("Индекс: %d чанков, %.1f МБ", collection.count(), index_size_mb())
    logger.info("Завершено за %.1f с", time.perf_counter() - started)
    logger.info(report.summary())
    return EXIT_DOCUMENTS_SKIPPED if report.errors else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
