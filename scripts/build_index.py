"""Строит векторный индекс Chroma.

Индекс базы знаний (knowledge_base/ -> chroma_db/): .venv/bin/python scripts/build_index.py
Индекс исходных текстов (raw/ -> chroma_db_raw/):  .venv/bin/python scripts/build_index.py raw
"""
import os
import re
import shutil
import sys
import time
from pathlib import Path

# Hugging Face работает без интернета: модель берется из локального кэша (скачивается командой bot/rag.py install).
# Нужно задать до импорта библиотек ниже.
os.environ["HF_HUB_OFFLINE"] = "1"

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "chroma_db"
RAW_INDEX_DIR = ROOT / "chroma_db_raw"
COLLECTION = "knowledge_base"
MODEL_NAME = "BAAI/bge-m3"
# Chroma по умолчанию отправляет в интернет статистику использования — отключаем
NO_TELEMETRY = Settings(anonymized_telemetry=False)

MIN_WORDS = 100
MAX_WORDS = 300

TOP_K = 3

# источник: (папка с текстами, папка индекса, пример запроса)
SOURCES = {
    "knowledge_base": (ROOT / "knowledge_base", INDEX_DIR, "Когда родился Тарен Сорвейн?"),
    "raw": (ROOT / "raw", RAW_INDEX_DIR, "Когда родился Люк Скайуокер?"),
}

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def count_words(texts):
    return sum(len(text.split()) for text in texts)


def read_sections(path):
    """Делит документ на заголовок статьи и разделы: [(название раздела, [абзацы])]."""
    blocks = [block.strip() for block in path.read_text(encoding="utf-8").split("\n\n") if block.strip()]
    title = blocks[0].lstrip("# ")
    sections = [("", [])]
    for block in blocks[1:]:
        if block.startswith("#"):
            sections.append((block.lstrip("# "), []))
        else:
            sections[-1][1].append(block)
    return title, [(heading, paragraphs) for heading, paragraphs in sections if paragraphs]


def split_long(paragraph):
    """Абзац длиннее MAX_WORDS режет по предложениям."""
    parts = []
    current = []
    for sentence in SENTENCE_END.split(paragraph):
        if current and count_words(current) + len(sentence.split()) > MAX_WORDS:
            parts.append(" ".join(current))
            current = []
        current.append(sentence)
    parts.append(" ".join(current))
    return parts


def make_chunks(paragraphs):
    """Склеивает соседние абзацы раздела, пока не наберется MIN_WORDS слов, но не больше MAX_WORDS."""
    pieces = [part for paragraph in paragraphs for part in split_long(paragraph)]
    chunks = []
    current = []
    for piece in pieces:
        if current and (count_words(current) >= MIN_WORDS or count_words(current) + len(piece.split()) > MAX_WORDS):
            chunks.append(current)
            current = []
        current.append(piece)
    chunks.append(current)
    # короткий хвост раздела приклеиваем к предыдущему чанку, если влезает
    if len(chunks) > 1 and count_words(chunks[-1]) < MIN_WORDS and count_words(chunks[-2] + chunks[-1]) <= MAX_WORDS:
        tail = chunks.pop()
        chunks[-1] += tail
    return ["\n\n".join(chunk) for chunk in chunks]


def build_chunks(docs_dir):
    """Возвращает id, тексты и метаданные всех чанков из папки с документами."""
    ids, texts, metadatas = [], [], []
    for path in sorted(docs_dir.glob("*.txt")):
        title, sections = read_sections(path)
        number = 0
        for heading, paragraphs in sections:
            prefix = f"{title}. {heading}." if heading else f"{title}."
            for chunk in make_chunks(paragraphs):
                ids.append(f"{path.name}#{number}")
                texts.append(f"{prefix}\n{chunk}")
                metadatas.append({
                    "source": str(path.relative_to(ROOT)),
                    "title": title,
                    "section": heading,
                    "chunk": number,
                })
                number += 1
    return ids, texts, metadatas


def create_collection(index_dir):
    """Строит индекс с нуля: старая папка удаляется целиком, чтобы не осталось старых чанков и файлов."""
    shutil.rmtree(index_dir, ignore_errors=True)
    client = chromadb.PersistentClient(path=str(index_dir), settings=NO_TELEMETRY)
    return client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def load_model():
    """Берет модель только из локального кэша — без обращений в интернет."""
    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except OSError as error:
        raise SystemExit(f"Модель {MODEL_NAME} не скачана: .venv/bin/python bot/rag.py install") from error


def show_example(model, collection, query):
    query_embedding = model.encode([query], normalize_embeddings=True)
    result = collection.query(query_embeddings=query_embedding.tolist(), n_results=TOP_K)
    print(f"\nПример запроса: {query}")
    for chunk_id, distance, text in zip(result["ids"][0], result["distances"][0], result["documents"][0]):
        print(f"\n[{chunk_id}] расстояние {distance:.3f}\n{text[:300]}...")


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "knowledge_base"
    if source not in SOURCES:
        sys.exit(f"Неизвестный источник {source}. Можно: {', '.join(SOURCES)}")
    docs_dir, index_dir, example_query = SOURCES[source]

    ids, texts, metadatas = build_chunks(docs_dir)
    documents = len({metadata["source"] for metadata in metadatas})
    words = [len(text.split()) for text in texts]
    print(f"Документов: {documents}, чанков: {len(ids)}, слов в чанке: {min(words)}-{max(words)}")

    model = load_model()
    started = time.perf_counter()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    elapsed = time.perf_counter() - started
    print(f"Эмбеддинги: {len(embeddings)} x {embeddings.shape[1]}, время генерации: {elapsed:.1f} с")

    collection = create_collection(index_dir)
    collection.add(ids=ids, documents=texts, embeddings=embeddings.tolist(), metadatas=metadatas)
    print(f"Индекс сохранен: {index_dir.relative_to(ROOT)}/, записей: {collection.count()}")

    show_example(model, collection, example_query)


if __name__ == "__main__":
    main()
