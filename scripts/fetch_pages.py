"""Скачивает статьи из русской Вукипедии и сохраняет чистый текст в raw/."""
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

API_URL = "https://starwars.fandom.com/ru/api.php"
RAW_DIR = Path(__file__).resolve().parent.parent / "raw"

MIN_WORDS = 500
MAX_WORDS = 1000
SHORT_WARNING_WORDS = 300
PAUSE_SECONDS = 1

STOP_SECTIONS = {
    "За кулисами", "Появления", "Неканоничные появления", "Источники",
    "Сноски и примечания", "Ссылки", "Внешние ссылки",
}
JUNK_SELECTORS = [
    "aside", "table", "figure", "sup", "dl", "style", "script",
    ".toc", ".mw-editsection", ".noprint",
]

PAGES = [
    # персонажи: пары герой — злодей
    "Люк Скайуокер", "Энакин Скайуокер",
    "Йода", "Палпатин",
    "Оби-Ван Кеноби", "Дарт Мол",
    "Лея Органа", "Уилхафф Таркин",
    "Хан Соло", "Боба Фетт",
    # объекты и технологии
    "Световой меч", "Бластер", "Гипердвигатель", "Звезда Смерти",
    "Тысячелетний сокол", "Раб I", "X-wing", "СИД-истребитель",
    # события
    "Приказ 66", "Дуэль на Мустафаре", "Уничтожение Алдераана",
    "Битва при Явине", "Битва при Хоте", "Битва при Эндоре",
    # стороны и места
    "Орден джедаев", "Орден ситхов", "Альянс повстанцев", "Галактическая Империя",
    "Сила", "Татуин", "Алдераан", "Корусант",
]

# Если имя файла не совпадает с названием статьи в Вукипедии
TITLE_OVERRIDES = {
    "X-wing": "Звёздный истребитель T-65 «X-wing»",
}

ENGLISH_NAME = re.compile(r"\s*\(англ\.[^)]*\)")


def fetch_html(title):
    params = {"action": "parse", "page": title, "prop": "text", "redirects": 1, "format": "json"}
    response = requests.get(API_URL, params=params, headers={"User-Agent": "rag-homework/1.0"}, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise ValueError(data["error"]["info"])
    return data["parse"]["text"]["*"]


def remove_junk(root):
    for element in root.select(", ".join(JUNK_SELECTORS)):
        element.decompose()


def collect_blocks(root):
    """Берет абзацы сверху вниз, пока не наберет MIN_WORDS слов, но не больше MAX_WORDS."""
    blocks = []
    words = 0
    pending_heading = None
    for element in root.find_all(["h2", "h3", "p"], recursive=False):
        text = " ".join(ENGLISH_NAME.sub("", element.get_text()).split())
        if not text:
            continue
        if element.name in ("h2", "h3"):
            if words >= MIN_WORDS or text in STOP_SECTIONS:
                break
            pending_heading = "#" * int(element.name[1]) + " " + text
            continue
        paragraph_words = len(text.split())
        if words >= MIN_WORDS or (words > 0 and words + paragraph_words > MAX_WORDS):
            break
        if pending_heading:
            blocks.append(pending_heading)
            pending_heading = None
        blocks.append(text)
        words += paragraph_words
    return blocks, words


def main():
    RAW_DIR.mkdir(exist_ok=True)
    for title in PAGES:
        try:
            html = fetch_html(TITLE_OVERRIDES.get(title, title))
        except (requests.RequestException, ValueError) as error:
            print(f"  ОШИБКА  {title}: {error}")
            continue
        root = BeautifulSoup(html, "html.parser").select_one(".mw-parser-output")
        remove_junk(root)
        blocks, words = collect_blocks(root)
        text = f"# {title}\n\n" + "\n\n".join(blocks) + "\n"
        (RAW_DIR / f"{title}.txt").write_text(text, encoding="utf-8")
        warning = "  <- мало" if words < SHORT_WARNING_WORDS else ""
        print(f"{words:5} слов  {title}{warning}")
        time.sleep(PAUSE_SECONDS)


if __name__ == "__main__":
    main()
