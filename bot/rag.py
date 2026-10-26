"""RAG: вопрос -> поиск чанков в Chroma -> промпт (few-shot + Chain-of-Thought) -> LLM -> ответ.

Скачать модель эмбеддингов (один раз): .venv/bin/python bot/rag.py install
Задать вопрос из терминала:            .venv/bin/python bot/rag.py [rag|raw|norag] "вопрос"
Режимы: rag — по базе знаний, raw — по исходным текстам из raw/, norag — без RAG, только LLM.
Настройки LLM — переменные окружения LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, LLM_REASONING_EFFORT.
Логи (LOG_LEVEL, по умолчанию INFO) показывают каждый шаг и запрос в LLM целиком.
Защита от промпт-инъекций (bot/guard.py): GUARD_LAYERS — номера включенных слоев, по умолчанию 1,2,3,4;
пустое значение — без защиты.
"""
import logging
import os
import re
import sys
import time
from pathlib import Path

# Hugging Face работает без интернета: модель берется из локального кэша.
# В сеть ходит только команда install. Нужно задать до импорта библиотек ниже.
if sys.argv[1:] != ["install"]:
    os.environ["HF_HUB_OFFLINE"] = "1"

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import guard

ROOT = Path(__file__).resolve().parent.parent
COLLECTION = "knowledge_base"
# режим: (папка индекса, команда, которая его строит)
INDEXES = {
    "rag": (ROOT / "chroma_db", ".venv/bin/python scripts/build_index.py"),
    "raw": (ROOT / "chroma_db_raw", ".venv/bin/python scripts/build_index.py raw"),
}
MODES = {
    "rag": "RAG по базе знаний",
    "raw": "RAG по исходным текстам (raw)",
    "norag": "без RAG — только LLM",
}
DEFAULT_MODE = "rag"
EMBEDDING_MODEL = "BAAI/bge-m3"
# Chroma по умолчанию отправляет в интернет статистику использования — отключаем
NO_TELEMETRY = Settings(anonymized_telemetry=False)

TOP_K = 3
MAX_DISTANCE = 0.5
TEMPERATURE = 0.2
I_DONT_KNOW = "Я не знаю — в базе знаний этого нет."

DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LLM_MODEL = "qwen3:8b"
DEFAULT_LLM_API_KEY = "ollama"
DEFAULT_REASONING_EFFORT = "none"

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

logger = logging.getLogger("rag")

SYSTEM_PROMPT = """Ты — помощник по базе знаний компании. Отвечай на русском языке.

Правила:
1. Отвечай только по фрагментам из базы знаний, которые даны в сообщении. Свои знания не используй.
2. Сначала по шагам разбери фрагменты: что в каком фрагменте сказано по вопросу. Потом дай короткий ответ.
3. Если во фрагментах нет ответа, напиши: "Ответ: Я не знаю — в базе знаний этого нет."
4. В конце укажи источник — файлы фрагментов, на которые опирался.

Формат:
Шаги:
1. ...
Ответ: ...
Источник: ..."""

NO_RAG_SYSTEM_PROMPT = "Ты — помощник. Отвечай на русском языке."

# Примеры для few-shot: у каждой базы свои, с ее названиями
FEW_SHOT = {
    "rag": [
        {
            "question": "Что такое Протокол 9?",
            "fragments": (
                "[Протокол 9.txt#0] Протокол 9 — один из чрезвычайных приказов Великой армии Лиги. "
                "Приказ объявлял сейраев изменниками Лиги и требовал их немедленного и беспрекословного уничтожения. "
                "Приказ был отдан Верховным канцлером Севраном и обозначил начало Великого истребления сейраев..."
            ),
            "answer": (
                "Шаги:\n"
                "1. Во фрагменте [Протокол 9.txt#0] сказано, что это чрезвычайный приказ Великой армии Лиги.\n"
                "2. Там же: приказ объявлял сейраев изменниками и требовал их уничтожить.\n"
                "3. Отдал приказ Верховный канцлер Севран.\n"
                "Ответ: Протокол 9 — чрезвычайный приказ Великой армии Лиги, по которому сейраев объявили "
                "изменниками и уничтожали. Его отдал канцлер Севран.\n"
                "Источник: Протокол 9.txt"
            ),
        },
        {
            "question": "Какой любимый цвет Неи Вирены?",
            "fragments": (
                "[Нея Вирена.txt#0] Нея Вирена-Ренто — дочь рыцаря-сейрая Ольвена Сорвейна и сенатора Илме Корвелы "
                "Тессари, а также сестра-близнец Тарена Сорвейна. После рождения ее удочерили Эдрик Вирена и королева "
                "Днеса, сделав ее принцессой Эльмарона..."
            ),
            "answer": (
                "Шаги:\n"
                "1. Во фрагменте [Нея Вирена.txt#0] есть ее семья и то, что она принцесса Эльмарона.\n"
                "2. О любимом цвете там ничего нет.\n"
                f"Ответ: {I_DONT_KNOW}"
            ),
        },
    ],
    "raw": [
        {
            "question": "Что такое Приказ 66?",
            "fragments": (
                "[Приказ 66.txt#0] Приказ 66 — один из чрезвычайных приказов Великой армии Республики. "
                "Приказ объявлял джедаев изменниками Республики и требовал их немедленного и беспрекословного уничтожения. "
                "Приказ был отдан Верховным канцлером Палпатином и обозначил начало Великого истребления джедаев..."
            ),
            "answer": (
                "Шаги:\n"
                "1. Во фрагменте [Приказ 66.txt#0] сказано, что это чрезвычайный приказ Великой армии Республики.\n"
                "2. Там же: приказ объявлял джедаев изменниками и требовал их уничтожить.\n"
                "3. Отдал приказ Верховный канцлер Палпатин.\n"
                "Ответ: Приказ 66 — чрезвычайный приказ Великой армии Республики, по которому джедаев объявили "
                "изменниками и уничтожали. Его отдал канцлер Палпатин.\n"
                "Источник: Приказ 66.txt"
            ),
        },
        {
            "question": "Какой любимый цвет Леи Органы?",
            "fragments": (
                "[Лея Органа.txt#0] Лея Органа-Соло — дочь рыцаря-джедая Энакина Скайуокера и сенатора Падме Амидалы "
                "Наберри, а также сестра-близнец Люка Скайуокера. После рождения ее удочерили Бейл Органа и королева "
                "Бреха, сделав ее принцессой Альдераана..."
            ),
            "answer": (
                "Шаги:\n"
                "1. Во фрагменте [Лея Органа.txt#0] есть ее семья и то, что она принцесса Альдераана.\n"
                "2. О любимом цвете там ничего нет.\n"
                f"Ответ: {I_DONT_KNOW}"
            ),
        },
    ],
}


def user_message(question, fragments):
    return f"Фрагменты:\n{fragments}\n\nВопрос: {question}"


def format_fragments(chunks):
    return "\n\n".join(f"[{chunk_id}] {text}" for chunk_id, _, text in chunks)


def build_messages(question, chunks, examples, system_prompt):
    """System-промпт, затем примеры (few-shot) как прошлые диалоги, затем настоящий вопрос."""
    messages = [{"role": "system", "content": system_prompt}]
    for example in examples:
        messages.append({"role": "user", "content": user_message(example["question"], example["fragments"])})
        messages.append({"role": "assistant", "content": example["answer"]})
    messages.append({"role": "user", "content": user_message(question, format_fragments(chunks))})
    return messages


def format_request(url, params, messages):
    """Запрос в LLM в читаемом виде: адрес, параметры и все сообщения по порядку."""
    lines = [f"POST {url}", "    " + ", ".join(f"{name}={value}" for name, value in params.items())]
    for number, message in enumerate(messages, start=1):
        lines.append(f"---- {number}. [{message['role']}] ----")
        lines.append(message["content"])
    lines.append("---- конец запроса ----")
    return "\n".join(lines)


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=os.environ.get("LOG_LEVEL", "INFO"),
    )
    # httpx пишет в лог адреса запросов, а у Телеграма в адресе есть токен — такие записи не выводим
    logging.getLogger("httpx").setLevel(logging.WARNING)


def read_guard_layers():
    """Номера слоев защиты из GUARD_LAYERS, например "1,2,3,4"."""
    value = os.environ.get("GUARD_LAYERS", ",".join(map(str, sorted(guard.ALL_LAYERS))))
    layers = frozenset(int(part) for part in value.split(",") if part.strip())
    if not layers <= guard.ALL_LAYERS:
        raise SystemExit(f"GUARD_LAYERS: допустимые слои {sorted(guard.ALL_LAYERS)}, получено {value}")
    return layers


def drop_secret_chunks(chunks):
    """Слой 2: выбрасывает чанки со словами про секреты."""
    kept = []
    for chunk in chunks:
        secrets = guard.find_secrets(chunk[2])
        if secrets:
            logger.warning("Защита, слой 2: выбрасываю чанк %s — признаки секретов: %s", chunk[0], secrets)
        else:
            kept.append(chunk)
    return kept


def remove_commands_from_chunks(chunks):
    """Слой 3: удаляет из чанков команды, обращенные к модели."""
    cleaned = []
    for chunk_id, distance, text in chunks:
        commands = guard.find_commands(text)
        if commands:
            logger.warning("Защита, слой 3: из чанка %s удаляю команды: %s", chunk_id, commands)
            text = guard.remove_commands(text)
        cleaned.append((chunk_id, distance, text))
    return cleaned


def install():
    """Скачивает модель эмбеддингов в кэш Hugging Face (~2,2 ГБ). Нужно один раз."""
    SentenceTransformer(EMBEDDING_MODEL)
    print(f"Модель {EMBEDDING_MODEL} скачана")


def load_embedder():
    """Берет модель только из локального кэша — без обращений в интернет."""
    try:
        return SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    except OSError as error:
        raise SystemExit(f"Модель {EMBEDDING_MODEL} не скачана: .venv/bin/python bot/rag.py install") from error


class Rag:
    def __init__(self):
        started = time.perf_counter()
        self.embedder = load_embedder()
        logger.info("Модель эмбеддингов %s загружена за %.1f с", EMBEDDING_MODEL, time.perf_counter() - started)
        self.collections = {}
        for mode, (index_dir, _) in INDEXES.items():
            # проверяем файл базы, а не папку: Docker создает пустую папку, если индекса нет
            if not (index_dir / "chroma.sqlite3").exists():
                logger.warning("Индекс %s не найден — режим «%s» недоступен", index_dir.name, MODES[mode])
                continue
            client = chromadb.PersistentClient(path=str(index_dir), settings=NO_TELEMETRY)
            self.collections[mode] = client.get_collection(COLLECTION)
            logger.info("Индекс %s: %d чанков (режим «%s»)", index_dir.name, self.collections[mode].count(), MODES[mode])
        base_url = os.environ.get("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
        self.llm = OpenAI(base_url=base_url, api_key=os.environ.get("LLM_API_KEY", DEFAULT_LLM_API_KEY))
        self.llm_model = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)
        self.reasoning_effort = os.environ.get("LLM_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
        logger.info("LLM: %s, модель %s, reasoning_effort=%s", base_url, self.llm_model, self.reasoning_effort or "-")
        self.guard_layers = read_guard_layers()
        logger.info("Слои защиты, когда она включена: %s", sorted(self.guard_layers) or "нет")

    def search(self, question, collection):
        """Ближайшие чанки: [(id, расстояние, текст)], самый близкий первым."""
        embedding = self.embedder.encode([question], normalize_embeddings=True).tolist()
        result = collection.query(query_embeddings=embedding, n_results=TOP_K)
        chunks = list(zip(result["ids"][0], result["distances"][0], result["documents"][0]))
        logger.info("1/4 Поиск: найдено чанков %d (расстояние, id):", len(chunks))
        for chunk_id, distance, _ in chunks:
            logger.info("      %.3f  %s", distance, chunk_id)
        return chunks

    def ask_llm(self, messages):
        # Пустое LLM_REASONING_EFFORT — параметр не отправляется (не все провайдеры его знают)
        options = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        params = {"model": self.llm_model, "temperature": TEMPERATURE, **options}
        url = f"{self.llm.base_url}chat/completions"
        logger.info("4/4 Запрос в LLM:\n%s", format_request(url, params, messages))
        started = time.perf_counter()
        response = self.llm.chat.completions.create(messages=messages, **params)
        usage = response.usage
        logger.info(
            "LLM ответила за %.1f с, токенов: %s на входе, %s на выходе",
            time.perf_counter() - started,
            usage.prompt_tokens if usage else "?",
            usage.completion_tokens if usage else "?",
        )
        return THINK_BLOCK.sub("", response.choices[0].message.content).strip()

    def answer(self, question, mode=DEFAULT_MODE, protected=True):
        layers = self.guard_layers if protected else frozenset()
        logger.info("Режим «%s», защита: %s. Вопрос: %s", MODES[mode], sorted(layers) or "выкл", question)
        if mode == "norag":
            reply = self.answer_without_rag(question)
        else:
            reply = self.answer_with_rag(question, mode, layers)
        if 4 in layers and (secrets := guard.find_secrets(reply)):
            logger.warning("Защита, слой 4: в ответе признаки секретов %s — вместо ответа отказ", secrets)
            logger.info("Ответ после защиты: %s", guard.REFUSAL)
            return guard.REFUSAL
        return reply

    def answer_with_rag(self, question, mode, layers):
        if mode not in self.collections:
            index_dir, build_command = INDEXES[mode]
            logger.warning("Индекс %s не построен", index_dir.name)
            return f"Индекс для режима «{MODES[mode]}» не построен. Команда: {build_command}"
        chunks = self.search(question, self.collections[mode])
        if 2 in layers:
            chunks = drop_secret_chunks(chunks)
        if not chunks or chunks[0][1] > MAX_DISTANCE:
            best = f"{chunks[0][1]:.3f}" if chunks else "нет чанков"
            logger.info("2/4 Порог: лучший чанк %s > %.2f — отвечаю «не знаю», LLM не вызываю", best, MAX_DISTANCE)
            logger.info("Ответ: %s", I_DONT_KNOW)
            return I_DONT_KNOW
        logger.info("2/4 Порог: лучший чанк %.3f <= %.2f — иду в LLM", chunks[0][1], MAX_DISTANCE)
        if 3 in layers:
            chunks = remove_commands_from_chunks(chunks)
        system_prompt = SYSTEM_PROMPT + guard.PROMPT_RULE if 1 in layers else SYSTEM_PROMPT
        messages = build_messages(question, chunks, FEW_SHOT[mode], system_prompt)
        prompt_chars = sum(len(message["content"]) for message in messages)
        logger.info(
            "3/4 Промпт: system + %d примера + вопрос с %d фрагментами, %d символов",
            len(FEW_SHOT[mode]), len(chunks), prompt_chars,
        )
        return self.ask_and_log(messages)

    def answer_without_rag(self, question):
        logger.info("Без RAG: поиск и промпт с фрагментами пропускаю, вопрос сразу в LLM")
        messages = [
            {"role": "system", "content": NO_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return self.ask_and_log(messages)

    def ask_and_log(self, messages):
        reply = self.ask_llm(messages)
        logger.info("Ответ (%d символов):\n%s", len(reply), reply)
        return reply


if __name__ == "__main__":
    setup_logging()
    args = sys.argv[1:]
    if args == ["install"]:
        install()
    elif args:
        mode = args.pop(0) if args[0] in MODES else DEFAULT_MODE
        Rag().answer(" ".join(args), mode)
    else:
        sys.exit('Укажи вопрос: .venv/bin/python bot/rag.py [rag|raw|norag] "вопрос" или install')
