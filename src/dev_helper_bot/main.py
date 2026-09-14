from __future__ import annotations

import asyncio
import io
import logging
from datetime import datetime

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message as TgMessage
from dotenv import load_dotenv

from dev_helper_bot.agent import run_agent
from dev_helper_bot.config import (
    embedding_dim,
    llm_model_name,
    make_embeddings,
    make_llm,
    memory_db_path,
    obs_db_path,
    obs_label,
    obs_price_input_per_m,
    obs_price_output_per_m,
    obs_web_host,
    obs_web_port,
    rag_db_path,
    rag_max_chunks_per_doc,
    rag_max_extract_chars,
    rag_max_upload_bytes,
    telegram_token,
)
from dev_helper_bot.document_store import DocumentStore, UserDocumentSearcher
from dev_helper_bot.documents import (
    SUPPORTED_EXTENSIONS,
    DocumentError,
    ensure_upload_size,
    is_supported_document,
    split_document,
)
from dev_helper_bot.embeddings import EmbeddingClient, EmbeddingsUnavailable
from dev_helper_bot.llm import LLMClient, LLMUnavailable, Message
from dev_helper_bot.memory import ChatHistorySearcher, MemoryStore
from dev_helper_bot.obs_web import build_app, start_app, stop_app
from dev_helper_bot.sandbox import SandboxExecutor, prepare_sandbox_environment
from dev_helper_bot.skills import Skill, build_request_messages, default_skills_dir, load_skills
from dev_helper_bot.telemetry import (
    RUN_STATUS_LLM_ERROR,
    ObservingClient,
    RunRecorder,
    TelemetryStore,
)
from dev_helper_bot.tools import (
    EXEC_TOOL_SPEC,
    GET_SKILL_TOOL_SPEC,
    LIST_TOOL_SPEC,
    READ_FILE_TOOL_SPEC,
    SEARCH_DOCUMENTS_TOOL_SPEC,
    SEARCH_TOOL_SPEC,
    CommandExecutor,
)

TELEGRAM_MESSAGE_LIMIT = 4096
WAITING_MESSAGE = "⏳ Готовлю ответ…"
NEW_CHAT_CONFIRMATION = "🆕 Контекст сброшен — начинаем новый диалог."

INDEXING_MESSAGE = "⏳ Обрабатываю «{filename}» — извлекаю текст и строю индекс…"
INDEXED_MESSAGE = (
    "✅ «{filename}» проиндексирован: фрагментов {count}. "
    "Можно задавать вопросы по документу."
)
DOCUMENT_ERROR_TEMPLATE = "⚠️ Не удалось проиндексировать «{filename}»: {reason}"
UNSUPPORTED_DOCUMENT_MESSAGE = (
    "⚠️ Формат файла «{filename}» не поддерживается. "
    f"Доступны: {', '.join(SUPPORTED_EXTENSIONS)}."
)
DOWNLOAD_ERROR_MESSAGE = (
    "⚠️ Не удалось скачать «{filename}» из Telegram. Попробуйте ещё раз."
)
EMBEDDINGS_ERROR_MESSAGE = (
    "⚠️ Сервис эмбеддингов недоступен — «{filename}» не проиндексирован. "
    "Попробуйте позже."
)
DOCUMENTS_DISABLED_MESSAGE = "⚠️ Индекс документов сейчас недоступен."
DOCUMENTS_EMPTY_MESSAGE = (
    "📄 Загруженных документов нет. Пришлите файл "
    f"({', '.join(SUPPORTED_EXTENSIONS)}), чтобы задавать вопросы по нему."
)
DOCUMENTS_LIST_HEADER = "📄 Ваши документы:"
DELETE_USAGE_MESSAGE = (
    "Использование: /delete <имя файла>. Список имён — /documents."
)
DELETE_DONE_TEMPLATE = "🗑 «{filename}» удалён из индекса."
DELETE_NOT_FOUND_TEMPLATE = (
    "Документ «{filename}» не найден среди ваших. Список — /documents."
)

AGENT_TOOLS = [
    EXEC_TOOL_SPEC,
    READ_FILE_TOOL_SPEC,
    GET_SKILL_TOOL_SPEC,
    SEARCH_TOOL_SPEC,
    LIST_TOOL_SPEC,
    SEARCH_DOCUMENTS_TOOL_SPEC,
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("bot")


async def send_chunked(bot: Bot, chat_id: int, text: str) -> None:
    for i in range(0, len(text), TELEGRAM_MESSAGE_LIMIT):
        chunk = text[i : i + TELEGRAM_MESSAGE_LIMIT]
        await bot.send_message(chat_id=chat_id, text=chunk)


async def handle_text(
    message: TgMessage,
    bot: Bot,
    llm: LLMClient,
    memory: MemoryStore,
    skills: dict[str, Skill],
    executor: CommandExecutor,
    telemetry: TelemetryStore | None = None,
    documents: DocumentStore | None = None,
    embeddings: EmbeddingClient | None = None,
) -> None:
    chat_id = message.chat.id
    user_text = message.text or ""
    # Канон — БД (design D4): контекст открытой сессии восстанавливается из
    # хранилища, транскрипт инструментов живёт только в рамках этой обработки.
    # Сборка запроса — единый шов skills.build_request_messages (design D4):
    # стабильный системный промпт + история + текущее сообщение с контекстной
    # строкой времени (в память не персистится).
    session_history = await memory.load_open_history(chat_id)
    history: list[Message] = build_request_messages(
        skills, session_history, user_text, datetime.now()
    )
    await memory.append_user(chat_id, user_text)

    # Телеметрия прогона (design D2): recorder на каждое сообщение, LLM —
    # в наблюдающей обёртке. Телеметрия best-effort и не влияет на ответы.
    recorder: RunRecorder | None = None
    client = llm
    if telemetry is not None:
        recorder = RunRecorder(
            telemetry,
            chat_id,
            obs_label(),
            price_input_per_m=obs_price_input_per_m(),
            price_output_per_m=obs_price_output_per_m(),
        )
        await recorder.start()
        client = ObservingClient(llm, recorder, model=llm_model_name())

    # Документы принадлежат отправителю, а не чату (design D3): в группе
    # у участников разные индексы, поэтому поиск строится на from_user.id.
    document_search = None
    if documents is not None and embeddings is not None:
        document_search = UserDocumentSearcher(
            documents, embeddings, _sender_id(message)
        )

    await bot.send_message(chat_id=chat_id, text=WAITING_MESSAGE)
    try:
        reply = await run_agent(
            client,
            history,
            tools=AGENT_TOOLS,
            executor=executor,
            history_search=ChatHistorySearcher(memory, chat_id),
            skills=skills,
            document_search=document_search,
            recorder=recorder,
        )
    except LLMUnavailable as exc:
        log.warning("LLM unavailable: %s", exc)
        # Ветка LLMUnavailable финализируется здесь (design D2): run_agent
        # терминальные возвраты закрывает сам, исключение проходит мимо.
        if recorder is not None:
            await recorder.finish(RUN_STATUS_LLM_ERROR)
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "⚠️ LLM сейчас недоступна. Проверьте, что сервер запущен, "
                "и попробуйте ещё раз."
            ),
        )
        return
    await memory.append_assistant(chat_id, reply)
    await send_chunked(bot, chat_id, reply)


async def handle_new(message: TgMessage, bot: Bot, memory: MemoryStore) -> None:
    await memory.close_session(message.chat.id)
    await bot.send_message(chat_id=message.chat.id, text=NEW_CHAT_CONFIRMATION)


def _sender_id(message: TgMessage) -> int:
    """Владелец документов — автор сообщения (design D3).

    Для чата без автора (канальные посты) владельцем остаётся сам чат:
    так документы не попадают в общий на всех бакет.
    """
    user = getattr(message, "from_user", None)
    return user.id if user is not None else message.chat.id


async def download_document(bot: Bot, document) -> bytes:
    """Скачивает файл Telegram в память: на диск он не кладётся (design: сырые
    файлы после индексации не храним)."""
    buffer = io.BytesIO()
    await bot.download(document, destination=buffer)
    return buffer.getvalue()


async def handle_document(
    message: TgMessage,
    bot: Bot,
    documents: DocumentStore | None = None,
    embeddings: EmbeddingClient | None = None,
) -> None:
    """Загрузка документа: проверки → скачивание → извлечение → chunking →
    эмбеддинги → индекс (design D6).

    Агентный цикл здесь не запускается: индексация не требует модели.
    Любой отказ — понятное сообщение в чат и отсутствие частичного индекса:
    запись в хранилище идёт одной транзакцией после успешных эмбеддингов.
    """
    chat_id = message.chat.id
    document = message.document
    filename = ((getattr(document, "file_name", None) or "").strip()) or "файл"

    if not is_supported_document(filename):
        await bot.send_message(
            chat_id=chat_id,
            text=UNSUPPORTED_DOCUMENT_MESSAGE.format(filename=filename),
        )
        return
    if documents is None or embeddings is None:
        await bot.send_message(chat_id=chat_id, text=DOCUMENTS_DISABLED_MESSAGE)
        return

    # Сырой размер — до скачивания и разбора (design D8).
    try:
        ensure_upload_size(getattr(document, "file_size", None), rag_max_upload_bytes())
    except DocumentError as exc:
        await bot.send_message(
            chat_id=chat_id,
            text=DOCUMENT_ERROR_TEMPLATE.format(filename=filename, reason=exc),
        )
        return

    await bot.send_message(
        chat_id=chat_id, text=INDEXING_MESSAGE.format(filename=filename)
    )
    try:
        data = await download_document(bot, document)
    except Exception:
        log.warning("cannot download document %s", filename, exc_info=True)
        await bot.send_message(
            chat_id=chat_id, text=DOWNLOAD_ERROR_MESSAGE.format(filename=filename)
        )
        return

    try:
        ensure_upload_size(len(data), rag_max_upload_bytes())
        chunks = split_document(
            filename,
            data,
            max_chars=rag_max_extract_chars(),
            max_chunks=rag_max_chunks_per_doc(),
        )
    except DocumentError as exc:
        await bot.send_message(
            chat_id=chat_id,
            text=DOCUMENT_ERROR_TEMPLATE.format(filename=filename, reason=exc),
        )
        return

    try:
        vectors = await embeddings.embed(chunks)
    except EmbeddingsUnavailable as exc:
        log.warning("embeddings unavailable while indexing %s: %s", filename, exc)
        await bot.send_message(
            chat_id=chat_id, text=EMBEDDINGS_ERROR_MESSAGE.format(filename=filename)
        )
        return

    count = await documents.index_document(
        _sender_id(message), filename, chunks, vectors
    )
    await bot.send_message(
        chat_id=chat_id,
        text=INDEXED_MESSAGE.format(filename=filename, count=count),
    )


async def handle_documents(
    message: TgMessage, bot: Bot, documents: DocumentStore | None = None
) -> None:
    """Команда /documents: список документов отправителя без вызова LLM."""
    chat_id = message.chat.id
    if documents is None:
        await bot.send_message(chat_id=chat_id, text=DOCUMENTS_DISABLED_MESSAGE)
        return
    infos = await documents.list_documents(_sender_id(message))
    if not infos:
        await bot.send_message(chat_id=chat_id, text=DOCUMENTS_EMPTY_MESSAGE)
        return
    lines = [DOCUMENTS_LIST_HEADER]
    for info in infos:
        lines.append(f"— {info.filename} (фрагментов: {info.chunk_count})")
    await bot.send_message(chat_id=chat_id, text="\n".join(lines))


async def handle_delete(
    message: TgMessage, bot: Bot, documents: DocumentStore | None = None
) -> None:
    """Команда /delete <имя файла>: удаление документа отправителя без LLM."""
    chat_id = message.chat.id
    # Аргумент — всё после самой команды; форма /delete@botname тоже работает.
    filename = (message.text or "").partition(" ")[2].strip()
    if not filename:
        await bot.send_message(chat_id=chat_id, text=DELETE_USAGE_MESSAGE)
        return
    if documents is None:
        await bot.send_message(chat_id=chat_id, text=DOCUMENTS_DISABLED_MESSAGE)
        return
    deleted = await documents.delete_by_filename(_sender_id(message), filename)
    text = (
        DELETE_DONE_TEMPLATE.format(filename=filename)
        if deleted
        else DELETE_NOT_FOUND_TEMPLATE.format(filename=filename)
    )
    await bot.send_message(chat_id=chat_id, text=text)


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
    memory = MemoryStore(memory_db_path())
    await memory.open()
    # Телеметрия — тот же паттерн владения, что у памяти (design D4);
    # сбой открытия не роняет бота: наблюдение не может стать причиной отказа.
    telemetry: TelemetryStore | None = TelemetryStore(obs_db_path())
    try:
        await telemetry.open()
    except Exception:
        log.warning("telemetry disabled: cannot open %s", obs_db_path(), exc_info=True)
        telemetry = None
    # Веб-дашборд (change add-obs-web-dashboard, design D1/D2): in-process
    # aiohttp на loopback. Сбой bind (занятый порт) — fail-fast: исключение
    # до polling с понятной причиной. Сбой телеметрии — soft: веб жив и отдаёт
    # заглушку «телеметрия недоступна». Цены для аудита — те же, что у recorder.
    # Весь lifecycle (веб + polling + ресурсы) — в одном try/finally, чтобы
    # при сбое bind порта закрыть memory/telemetry, а не оставить открытыми.
    web_app = build_app(
        telemetry,
        price_input_per_m=obs_price_input_per_m(),
        price_output_per_m=obs_price_output_per_m(),
    )
    web_runner = None
    # Индекс документов открывается внутри try, чтобы его fail-fast (нет
    # расширения sqlite-vec — design D10) не оставил память и телеметрию
    # открытыми. RAG обязателен: молча работать без документов нечестно.
    documents = DocumentStore(rag_db_path(), embedding_dim())
    try:
        await documents.open()
        web_runner = await start_app(web_app, host=obs_web_host(), port=obs_web_port())
        dp = Dispatcher()
        dp["llm"] = llm
        dp["memory"] = memory
        dp["telemetry"] = telemetry
        dp["skills"] = load_skills(default_skills_dir())
        dp["executor"] = executor
        dp["documents"] = documents
        dp["embeddings"] = make_embeddings()
        # Команды и документы разбираются до текстового обработчика: иначе
        # /documents и /delete уехали бы в агентный цикл обычным текстом.
        dp.message.register(handle_new, Command("new"))
        dp.message.register(handle_documents, Command("documents"))
        dp.message.register(handle_delete, Command("delete"))
        dp.message.register(handle_document, F.document)
        dp.message.register(handle_text, F.text)

        log.info("Bot started. Long-polling…")
        await dp.start_polling(bot)
    finally:
        # Контейнер-жильца убираем best-effort: ошибки удаления не должны
        # прерывать завершение (спека docker-sandbox). Оставшийся после
        # аварийного завершения контейнер подберёт sweep при следующем старте.
        if web_runner is not None:
            await stop_app(web_runner)
        await executor.stop()
        await memory.close()
        await documents.close()
        if telemetry is not None:
            await telemetry.close()
        await bot.session.close()


def cli() -> None:
    """Synchronous entry point for the `dev-helper-bot` console script."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
