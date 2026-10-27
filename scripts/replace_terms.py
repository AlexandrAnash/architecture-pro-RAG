"""Заменяет термины по словарю terms_map.json: тексты из raw/ -> knowledge_base/.

Ключ словаря — исходное слово, значение — вымышленное. Окончание переносится
с исходного слова на новое: "Люка" -> "Тарена", "Люком" -> "Тареном".
Если основа слова короче словарной формы, граница отмечена "|": "Йод|а" -> "Морв|а".
Все замены делаются за один проход по тексту, длинные термины проверяются первыми.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw"
KB_DIR = ROOT / "knowledge_base"
TERMS_FILE = ROOT / "terms_map.json"

ENDINGS = [
    "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими",
    "ой", "ей", "ою", "ом", "ем", "ам", "ям", "ах", "ях", "ов", "ев",
    "ий", "ый", "ая", "яя", "ое", "ее", "ую", "юю", "ые", "ие", "ых", "их", "ым", "им",
    "а", "я", "у", "ю", "е", "и", "ы", "о", "й", "ь", "",
]
ENDING_PATTERN = "|".join(ENDINGS)
STRESS_MARK = "́"


def stems(phrase):
    """'Звезд|а Смерти' -> ['Звезд', 'Смерти']"""
    return [word.split("|")[0] for word in phrase.split(" ")]


def with_capitalized(terms):
    """Для терминов со строчной буквы добавляет вариант с заглавной: джедай -> Джедай."""
    result = dict(terms)
    for source, target in terms.items():
        if source[0].islower():
            result[source[0].upper() + source[1:]] = target[0].upper() + target[1:]
    return result


def build_replacer(terms):
    """Собирает одно регулярное выражение на все термины и функцию замены для него."""
    entries = sorted(terms.items(), key=lambda item: len(item[0]), reverse=True)
    patterns = []
    targets = []
    for index, (source, target) in enumerate(entries):
        source_stems, target_stems = stems(source), stems(target)
        if len(source_stems) != len(target_stems):
            raise ValueError(f"Разное число слов: {source} -> {target}")
        words = [f"{re.escape(stem)}(?P<t{index}e{n}>{ENDING_PATTERN})" for n, stem in enumerate(source_stems)]
        patterns.append(f"(?P<t{index}>{' '.join(words)})")
        targets.append(target_stems)
    regex = re.compile(r"(?<!\w)(?:" + "|".join(patterns) + r")(?!\w)")

    def replace(match):
        index = int(match.lastgroup[1:])
        return " ".join(stem + match.group(f"t{index}e{n}") for n, stem in enumerate(targets[index]))

    return regex, replace


def normalize(text):
    """ё -> е и без знаков ударения: "Палпати́н" -> "Палпатин". Так словарю хватает одного написания."""
    return text.replace("ё", "е").replace("Ё", "Е").replace(STRESS_MARK, "")


def main():
    terms = json.loads(TERMS_FILE.read_text(encoding="utf-8"))
    regex, replace = build_replacer(with_capitalized(terms))
    KB_DIR.mkdir(exist_ok=True)
    total = 0
    leftovers = 0
    for path in sorted(RAW_DIR.glob("*.txt")):
        text = normalize(path.read_text(encoding="utf-8"))
        new_text, count = regex.subn(replace, text)
        title = new_text.splitlines()[0].removeprefix("# ")
        (KB_DIR / f"{title}.txt").write_text(new_text, encoding="utf-8")
        total += count
        leftovers += len(regex.findall(new_text))
        print(f"{count:4} замен  {path.stem} -> {title}")
    print(f"Всего замен: {total}. Осталось терминов из словаря: {leftovers}")


if __name__ == "__main__":
    main()
