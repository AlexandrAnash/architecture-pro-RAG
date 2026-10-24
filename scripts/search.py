"""Поиск по индексу: для каждого вопроса показывает ближайшие чанки и расстояние до них.

Запуск: .venv/bin/python scripts/search.py "вопрос 1" "вопрос 2"
"""
import os
import sys

# Hugging Face работает без интернета: модель берется из локального кэша. Нужно задать до импорта библиотек ниже.
os.environ["HF_HUB_OFFLINE"] = "1"

import chromadb

from build_index import COLLECTION, INDEX_DIR, NO_TELEMETRY, TOP_K, load_model

PREVIEW_CHARS = 100


def main():
    questions = sys.argv[1:]
    if not questions:
        sys.exit('Укажи вопрос: .venv/bin/python scripts/search.py "вопрос"')
    model = load_model()
    collection = chromadb.PersistentClient(path=str(INDEX_DIR), settings=NO_TELEMETRY).get_collection(COLLECTION)
    for question in questions:
        embedding = model.encode([question], normalize_embeddings=True).tolist()
        result = collection.query(query_embeddings=embedding, n_results=TOP_K)
        print(f"\n{question}")
        for chunk_id, distance, text in zip(result["ids"][0], result["distances"][0], result["documents"][0]):
            preview = " ".join(text.split())[:PREVIEW_CHARS]
            print(f"  {distance:.3f}  [{chunk_id}] {preview}...")


if __name__ == "__main__":
    main()
