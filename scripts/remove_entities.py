"""Вносит в базу знаний искусственные пробелы: удаляет статьи сущностей и абзацы, где они упоминаются.

Запуск: .venv/bin/python scripts/remove_entities.py
После него индекс обновляется как обычно: scripts/update.sh
Вернуть базу: git checkout knowledge_base/
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "knowledge_base"

# Сущность (= имя файла статьи) и шаблон ее упоминаний: основа слова, чтобы найти все падежи
ENTITIES = {
    "Мегарис": r"мегарис",
    "Варда Кесс": r"\bвард(а|ы|е|у|ой|ою)\b",
    "Импульсер": r"импульсер",
}
MENTION = re.compile("|".join(ENTITIES.values()), re.IGNORECASE)
BLOCK_SEPARATOR = "\n\n"


def delete_articles():
    for name in ENTITIES:
        article = DOCS_DIR / f"{name}.txt"
        if article.exists():
            article.unlink()
            print(f"Удалена статья: {article.name}")


def remove_mentions(path):
    """Удаляет абзацы и заголовки разделов с упоминаниями. Заголовок статьи (первый блок) остается. Возвращает, сколько удалено."""
    title, *blocks = path.read_text(encoding="utf-8").split(BLOCK_SEPARATOR)
    kept = [block for block in blocks if not MENTION.search(block)]
    removed = len(blocks) - len(kept)
    if removed:
        path.write_text(BLOCK_SEPARATOR.join([title, *kept]), encoding="utf-8")
    return removed


def main():
    delete_articles()
    total = 0
    for path in sorted(DOCS_DIR.glob("*.txt")):
        removed = remove_mentions(path)
        if removed:
            print(f"Удалено абзацев и заголовков с упоминаниями: {removed:2}  {path.name}")
            total += removed
    left = sum(len(MENTION.findall(path.read_text(encoding="utf-8"))) for path in DOCS_DIR.glob("*.txt"))
    print(f"Всего удалено абзацев и заголовков: {total}. Осталось упоминаний: {left}")


if __name__ == "__main__":
    main()
