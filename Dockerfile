FROM python:3.11-slim

# бот работает не от root
RUN useradd --create-home --uid 1000 bot
WORKDIR /app

# torch только для CPU: обычная сборка для Linux тянет библиотеки NVIDIA на несколько гигабайт
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot/ bot/

# Модель эмбеддингов лежит в томе hf_cache, индекс — в подключенной папке chroma_db/ (см. docker-compose.yml).
# Папки кэша и лога запросов создаем заранее, чтобы бот мог в них писать.
ENV HF_HOME=/home/bot/.cache/huggingface \
    PYTHONUNBUFFERED=1
RUN mkdir -p "$HF_HOME" /app/logs && chown -R bot:bot /home/bot/.cache /app/logs

USER bot
CMD ["python", "bot/telegram_bot.py"]
