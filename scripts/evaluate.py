"""Автоматическая проверка качества RAG-бота на золотом наборе вопросов (golden_questions.json).

Задает вопросы по очереди через тот же Rag, что и бот,
поэтому каждый запрос попадает и в общий лог запросов logs/queries.jsonl.
Оценка каждого ответа сохраняется в logs/evaluation_<метка>.jsonl, сводка выводится на экран.
Запуск: .venv/bin/python scripts/evaluate.py [метка прогона]   например: evaluate.py full, потом evaluate.py gaps
Код выхода 1 — есть вопросы с оценкой не ok.
"""
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bot"))

from rag import Answer, Rag, Status  # noqa: E402 — путь к bot/ добавлен строкой выше

GOLDEN_FILE = ROOT / "golden_questions.json"
RESULTS_DIR = ROOT / "logs"
KEYWORD_VARIANTS = "|"
# Итоговая часть ответа: ключевые слова ищем только в ней, чтобы не засчитывать названия файлов из шагов и источников
FINAL_ANSWER = re.compile(r"Ответ:(.*?)(?:Источник:|$)", re.DOTALL)


class Verdict(StrEnum):
    OK = "ok"  # ответ полный, или темы в базе нет и бот честно отказался
    PARTIAL = "partial"  # ответ есть, но найдены не все ключевые слова
    GAP = "gap"  # тема должна быть в базе, а бот не знает — пробел в базе
    HALLUCINATION = "hallucination"  # темы в базе нет, а бот ответил
    FILTERED = "filtered"  # защита заменила ответ отказом
    ERROR = "error"  # сбой или не построен индекс


@dataclass
class GoldenQuestion:
    question: str
    group: str  # kept, removed или absent — описание групп в golden_questions.json
    answerable: bool
    expected_answer: str
    keywords: list
    source: str | None


def load_golden(path):
    return [GoldenQuestion(**item) for item in json.loads(path.read_text(encoding="utf-8"))["questions"]]


def final_answer(text):
    match = FINAL_ANSWER.search(text)
    return match.group(1) if match else text


def has_keyword(text, keyword):
    """Ключевое слово с вариантами через |: "Севран|Мальгор" засчитывается, если в тексте есть любой из них."""
    return any(variant.strip().lower() in text for variant in keyword.split(KEYWORD_VARIANTS))


def judge(golden, answer, found_keywords):
    if answer.status in (Status.ERROR, Status.NO_INDEX):
        return Verdict.ERROR
    if answer.status == Status.FILTERED:
        return Verdict.FILTERED
    answered = answer.status == Status.ANSWERED
    if not golden.answerable:
        return Verdict.HALLUCINATION if answered else Verdict.OK
    if not answered:
        return Verdict.GAP
    return Verdict.OK if len(found_keywords) == len(golden.keywords) else Verdict.PARTIAL


def ask(rag, question):
    """Ответ бота; сбой LLM не останавливает прогон, а становится ответом со статусом error."""
    try:
        return rag.answer(question)
    except Exception as error:
        return Answer(f"Ошибка: {error}", Status.ERROR)


def evaluate(rag, golden):
    started = time.perf_counter()
    answer = ask(rag, golden.question)
    answer_text = final_answer(answer.text).lower()
    found = [keyword for keyword in golden.keywords if has_keyword(answer_text, keyword)]
    found_files = {source.split("#")[0] for source in answer.sources}
    return {
        "question": golden.question,
        "group": golden.group,
        "expected_answer": golden.expected_answer,
        "verdict": judge(golden, answer, found),
        "status": answer.status,
        "completeness": round(len(found) / len(golden.keywords), 2) if golden.keywords else None,
        "keywords_found": found,
        "keywords_missing": [keyword for keyword in golden.keywords if keyword not in found],
        "expected_source": golden.source or None,
        "sources": answer.sources,
        # None — сравнивать не с чем: источник не ожидается или чанков не нашлось вовсе
        "relevant_source": golden.source in found_files if golden.source and found_files else None,
        "best_distance": answer.log_fields()["best_distance"],
        "answer_length": len(answer.text),
        "seconds": round(time.perf_counter() - started, 1),
        "answer": answer.text,
    }


def format_row(number, total, result):
    completeness = "-" if result["completeness"] is None else f"{result['completeness']:.0%}"
    source = {True: "да", False: "НЕТ", None: "-"}[result["relevant_source"]]
    distance = "-" if result["best_distance"] is None else f"{result['best_distance']:.3f}"
    return (
        f"[{number:2}/{total}] {result['group']:<7} {result['verdict']:<13} полнота {completeness:>4}  источник {source:<3}  "
        f"расстояние {distance}  {result['seconds']:5.1f} с  {result['question']}"
    )


def print_summary(results, results_file):
    verdicts = Counter(result["verdict"] for result in results)
    print("\nИтог:", ", ".join(f"{verdict} {count}" for verdict, count in verdicts.most_common()))
    gaps = [result["question"] for result in results if result["verdict"] == Verdict.GAP]
    print(f"Пробелов в базе: {len(gaps)}", *(f"  - {question}" for question in gaps), sep="\n")
    wrong_sources = [result["question"] for result in results if result["relevant_source"] is False]
    print(f"Нет ожидаемого источника среди найденных: {len(wrong_sources)}", *(f"  - {q}" for q in wrong_sources), sep="\n")
    print(f"Подробности: {results_file.relative_to(ROOT)}")


def main():
    # Логи RAG по каждому шагу здесь мешают читать сводку — показываем только предупреждения
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", level=os.environ.get("LOG_LEVEL", "WARNING"))
    label = sys.argv[1] if len(sys.argv) > 1 else f"{datetime.now():%Y-%m-%d_%H-%M}"
    results_file = RESULTS_DIR / f"evaluation_{label}.jsonl"
    golden_questions = load_golden(GOLDEN_FILE)
    print(f"Прогон «{label}»: {len(golden_questions)} вопросов из {GOLDEN_FILE.name}")
    rag = Rag()
    results = []
    RESULTS_DIR.mkdir(exist_ok=True)
    with results_file.open("w", encoding="utf-8") as file:
        for number, golden in enumerate(golden_questions, start=1):
            result = evaluate(rag, golden)
            results.append(result)
            file.write(json.dumps({"run": label, **result}, ensure_ascii=False) + "\n")
            print(format_row(number, len(golden_questions), result), flush=True)
    print_summary(results, results_file)
    return 0 if all(result["verdict"] == Verdict.OK for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
