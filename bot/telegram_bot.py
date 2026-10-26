"""Telegram-бот: принимает вопрос и отвечает через RAG.

Токен берется из переменной окружения TELEGRAM_TOKEN.
Команды: /rag — по базе знаний, /raw — по исходным текстам, /norag — без RAG, /mode — текущий режим,
/protect — включить или выключить защиту от промпт-инъекций. Режим и защита запоминаются для каждого чата.
Запуск: .venv/bin/python bot/telegram_bot.py
"""
import asyncio
import logging
import os
import time

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from rag import DEFAULT_MODE, MODES, Rag, setup_logging

GREETING = (
    "Привет! Я отвечаю на вопросы по базе знаний. Например: «Что такое Протокол 9?»\n\n"
    "Режимы: /rag — по базе знаний (по умолчанию), /raw — по исходным текстам, /norag — без RAG.\n"
    "/mode — какой режим сейчас. /protect — включить или выключить защиту от промпт-инъекций."
)
ERROR_REPLY = "Не получилось получить ответ. Попробуй еще раз чуть позже."

logger = logging.getLogger("bot")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("/start из чата %s", update.effective_chat.id)
    await update.message.reply_text(GREETING)


def current_mode(context):
    return context.chat_data.get("mode", DEFAULT_MODE)


def is_protected(context):
    return context.chat_data.get("protected", True)


def protection_label(context):
    return "вкл" if is_protected(context) else "выкл"


async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # "/raw" или "/raw@имя_бота" -> "raw"
    mode = update.message.text.split()[0].lstrip("/").split("@")[0]
    context.chat_data["mode"] = mode
    logger.info("Чат %s: режим «%s»", update.effective_chat.id, MODES[mode])
    await update.message.reply_text(f"Режим: {MODES[mode]}")


async def show_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"Сейчас режим: {MODES[current_mode(context)]}, защита: {protection_label(context)}\n"
        "Переключить: /rag, /raw, /norag, /protect"
    )


async def toggle_protection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.chat_data["protected"] = not is_protected(context)
    logger.info("Чат %s: защита %s", update.effective_chat.id, protection_label(context))
    await update.message.reply_text(f"Защита от промпт-инъекций: {protection_label(context)}")


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    mode = current_mode(context)
    protected = is_protected(context)
    logger.info(
        "Сообщение из чата %s (режим «%s», защита: %s): %s",
        update.effective_chat.id, MODES[mode], protection_label(context), question,
    )
    started = time.perf_counter()
    await update.message.chat.send_action("typing")
    try:
        # RAG работает долго и не умеет async — запускаем его в отдельном потоке, чтобы бот не зависал
        reply = await asyncio.to_thread(context.bot_data["rag"].answer, question, mode, protected)
    except Exception:
        logger.exception("Ошибка при ответе на вопрос")
        reply = ERROR_REPLY
    await update.message.reply_text(f"[{MODES[mode]}, защита: {protection_label(context)}]\n{reply}")
    logger.info("Ответ отправлен в чат %s, всего %.1f с", update.effective_chat.id, time.perf_counter() - started)


async def register_commands(app):
    """Команды появятся в меню бота в Телеграме."""
    commands = [BotCommand(mode, description) for mode, description in MODES.items()]
    commands.append(BotCommand("mode", "какой режим сейчас"))
    commands.append(BotCommand("protect", "включить или выключить защиту от промпт-инъекций"))
    await app.bot.set_my_commands(commands)


def main():
    setup_logging()
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise SystemExit("Нет токена: задай переменную окружения TELEGRAM_TOKEN")
    app = Application.builder().token(token).post_init(register_commands).build()
    app.bot_data["rag"] = Rag()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler(list(MODES), set_mode))
    app.add_handler(CommandHandler("mode", show_mode))
    app.add_handler(CommandHandler("protect", toggle_protection))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    logger.info("Бот запущен, жду сообщения")
    app.run_polling()


if __name__ == "__main__":
    main()
