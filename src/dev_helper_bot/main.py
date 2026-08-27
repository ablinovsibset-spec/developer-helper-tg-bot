from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import Message as TgMessage
from dotenv import load_dotenv

from dev_helper_bot.config import make_llm, telegram_token
from dev_helper_bot.llm import LLMClient, LLMUnavailable, Message

TELEGRAM_MESSAGE_LIMIT = 4096

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("bot")


async def send_chunked(bot: Bot, chat_id: int, text: str) -> None:
    for i in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        chunk = text[i : i + TELEGRAM_MESSAGE_LIMIT]
        await bot.send_message(chat_id=chat_id, text=chunk)


async def handle_text(message: TgMessage, bot: Bot, llm: LLMClient) -> None:
    messages: list[Message] = [{"role": "user", "content": message.text or ""}]
    try:
        reply = await llm.complete(messages)
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


async def main() -> None:
    load_dotenv()
    token = telegram_token()
    llm = make_llm()

    bot = Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp["llm"] = llm
    dp.message.register(handle_text, F.text)

    log.info("Bot started. Long-polling…")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())


def cli() -> None:
    """Synchronous entry point for the `dev-helper-bot` console script."""
    asyncio.run(main())
