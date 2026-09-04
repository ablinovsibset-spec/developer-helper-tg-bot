from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message as TgMessage
from dotenv import load_dotenv

from dev_helper_bot.agent import run_agent
from dev_helper_bot.config import make_llm, telegram_token
from dev_helper_bot.llm import LLMClient, LLMUnavailable, Message
from dev_helper_bot.sandbox import SandboxExecutor, prepare_sandbox_environment
from dev_helper_bot.skills import build_system_prompt, default_skills_dir, load_skills
from dev_helper_bot.tools import EXEC_TOOL_SPEC, CommandExecutor

TELEGRAM_MESSAGE_LIMIT = 4096
WAITING_MESSAGE = "⏳ Готовлю ответ…"
NEW_CHAT_CONFIRMATION = "🆕 Контекст сброшен — начинаем новый диалог."

ChatHistories = dict[int, list[Message]]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("bot")


async def send_chunked(bot: Bot, chat_id: int, text: str) -> None:
    for i in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        chunk = text[i : i + TELEGRAM_MESSAGE_LIMIT]
        await bot.send_message(chat_id=chat_id, text=chunk)


def chat_history(
    histories: ChatHistories, chat_id: int, system_prompt: str
) -> list[Message]:
    """История чата; системное сообщение обновляется актуальным промптом."""
    history = histories.get(chat_id)
    if not history:
        history = [{"role": "system", "content": system_prompt}]
        histories[chat_id] = history
    else:
        history[0]["content"] = system_prompt
    return history


async def handle_text(
    message: TgMessage,
    bot: Bot,
    llm: LLMClient,
    histories: ChatHistories,
    skills: dict[str, str],
    executor: CommandExecutor,
) -> None:
    system_prompt = build_system_prompt(skills, datetime.now())
    history = chat_history(histories, message.chat.id, system_prompt)
    history.append({"role": "user", "content": message.text or ""})

    await bot.send_message(chat_id=message.chat.id, text=WAITING_MESSAGE)
    try:
        reply = await run_agent(
            llm, history, tools=[EXEC_TOOL_SPEC], executor=executor
        )
    except LLMUnavailable as exc:
        log.warning("LLM unavailable: %s", exc)
        await bot.send_message(
            chat_id=message.chat.id,
            text=(
                "⚠️ LLM сейчас недоступна. Проверьте, что сервер запущен, "
                "и попробуйте ещё раз."
            ),
        )
        return
    await send_chunked(bot, message.chat.id, reply)


async def handle_new(
    message: TgMessage, bot: Bot, histories: ChatHistories
) -> None:
    histories[message.chat.id] = []
    await bot.send_message(chat_id=message.chat.id, text=NEW_CHAT_CONFIRMATION)


async def main() -> None:
    load_dotenv()
    token = telegram_token()
    llm = make_llm()

    bot = Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    await prepare_sandbox_environment()
    executor = SandboxExecutor()
    dp = Dispatcher()
    dp["llm"] = llm
    dp["histories"] = {}
    dp["skills"] = load_skills(default_skills_dir())
    dp["executor"] = executor
    dp.message.register(handle_new, Command("new"))
    dp.message.register(handle_text, F.text)

    log.info("Bot started. Long-polling…")
    try:
        await dp.start_polling(bot)
    finally:
        # Контейнер-жильца убираем best-effort: ошибки удаления не должны
        # прерывать завершение (спека docker-sandbox). Оставшийся после
        # аварийного завершения контейнер подберёт sweep при следующем старте.
        await executor.stop()
        await bot.session.close()


def cli() -> None:
    """Synchronous entry point for the `dev-helper-bot` console script."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
