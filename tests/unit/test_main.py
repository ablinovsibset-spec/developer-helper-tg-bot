from __future__ import annotations

from datetime import datetime

import pytest

from dev_helper_bot import main as main_module
from dev_helper_bot.agent import STEPS_EXHAUSTED_MESSAGE
from dev_helper_bot.llm import LLMUnavailable
from dev_helper_bot.main import (
    AGENT_TOOLS,
    NEW_CHAT_CONFIRMATION,
    TELEGRAM_MESSAGE_LIMIT,
    WAITING_MESSAGE,
    handle_new,
    handle_text,
    send_chunked,
)
from dev_helper_bot.memory import MemoryStore
from dev_helper_bot.skills import SANDBOX_ENV_LINE

from tests.conftest import (
    FakeCommandExecutor,
    FakeMessage,
    assistant_turn,
    make_llm_stub,
    make_scripted_llm,
    tool_call,
)

CHAT_ID = 42
SKILLS = {"wttr-in-api": "Правила wttr.in"}
T1 = datetime(2026, 8, 28, 7, 45)
T2 = datetime(2026, 8, 28, 7, 47)
SYSTEM_AT_T1 = (
    "Reasoning: medium\nТекущие дата и время: 2026-08-28 07:45 (пятница)"
    f"\n{SANDBOX_ENV_LINE}"
    "\n\n## wttr-in-api\nПравила wttr.in"
)
SYSTEM_AT_T2 = (
    "Reasoning: medium\nТекущие дата и время: 2026-08-28 07:47 (пятница)"
    f"\n{SANDBOX_ENV_LINE}"
    "\n\n## wttr-in-api\nПравила wttr.in"
)


def fake_datetime(*times: datetime):
    """datetime-заглушка: now() по очереди возвращает заданные моменты."""

    class FakeDatetime(datetime):
        _times = list(times)

        @classmethod
        def now(cls, tz=None):
            return cls._times.pop(0)

    return FakeDatetime


@pytest.fixture
async def store(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    await store.open()
    yield store
    await store.close()


async def handle(
    message,
    fake_bot,
    llm,
    store,
    skills=SKILLS,
    executor=None,
):
    executor = executor if executor is not None else FakeCommandExecutor()
    await handle_text(message, fake_bot, llm, store, skills, executor)


async def test_handle_text_sends_waiting_then_final_reply(fake_bot, store):
    llm = make_llm_stub(reply="ответ модели")

    await handle(FakeMessage("привет", chat_id=CHAT_ID), fake_bot, llm, store)

    assert fake_bot.sent == [
        {"chat_id": CHAT_ID, "text": WAITING_MESSAGE},
        {"chat_id": CHAT_ID, "text": "ответ модели"},
    ]


async def test_handle_text_prompt_goes_to_llm_with_system_and_tools(
    fake_bot, store, monkeypatch
):
    llm = make_llm_stub(reply="ok")
    monkeypatch.setattr(main_module, "datetime", fake_datetime(T1))

    await handle(FakeMessage("привет", chat_id=CHAT_ID), fake_bot, llm, store)

    assert llm.requests[0] == [
        {"role": "system", "content": SYSTEM_AT_T1},
        {"role": "user", "content": "привет"},
    ]
    assert llm.tools_per_request == [AGENT_TOOLS]


async def test_handle_text_llm_unavailable_sends_friendly_error(fake_bot, store):
    llm = make_llm_stub(error=LLMUnavailable("connection refused"))

    await handle(FakeMessage("привет", chat_id=CHAT_ID), fake_bot, llm, store)

    assert len(fake_bot.sent) == 2  # ⏳ и сообщение об ошибке
    assert fake_bot.sent[-1]["chat_id"] == CHAT_ID
    assert "недоступна" in fake_bot.sent[-1]["text"]
    # Ошибка LLM оставляет user-сообщение без ответа (design D4)
    assert await store.load_open_history(CHAT_ID) == [
        {"role": "user", "content": "привет"}
    ]


async def test_context_is_kept_between_messages(fake_bot, store, monkeypatch):
    llm = make_llm_stub(reply="первый ответ")
    monkeypatch.setattr(main_module, "datetime", fake_datetime(T1, T2))
    await handle(
        FakeMessage("первое", chat_id=CHAT_ID), fake_bot, llm, store
    )

    llm.turns = [assistant_turn("второй ответ")]
    await handle(FakeMessage("второе", chat_id=CHAT_ID), fake_bot, llm, store)

    assert llm.requests[1] == [
        {"role": "system", "content": SYSTEM_AT_T2},
        {"role": "user", "content": "первое"},
        {"role": "assistant", "content": "первый ответ"},
        {"role": "user", "content": "второе"},
    ]


async def test_system_message_is_refreshed_between_messages(
    fake_bot, store, monkeypatch
):
    llm = make_llm_stub(reply="первый ответ")
    monkeypatch.setattr(main_module, "datetime", fake_datetime(T1, T2))
    await handle(
        FakeMessage("первое", chat_id=CHAT_ID), fake_bot, llm, store
    )

    llm.turns = [assistant_turn("второй ответ")]
    await handle(FakeMessage("второе", chat_id=CHAT_ID), fake_bot, llm, store)

    assert llm.requests[0][0] == {"role": "system", "content": SYSTEM_AT_T1}
    assert llm.requests[1][0] == {"role": "system", "content": SYSTEM_AT_T2}
    assert [m["role"] for m in llm.requests[1]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]


async def test_tool_transcript_in_flight_but_not_persisted(fake_bot, store):
    """Транскрипт виден внутри обработки сообщения, но в БД попадают
    только исходное user-сообщение и финальный ответ."""
    llm = make_scripted_llm(
        [
            assistant_turn(
                content=None,
                tool_calls=[tool_call(arguments='{"command": "echo hi"}')],
                finish_reason="tool_calls",
            ),
            assistant_turn(content="готово"),
        ]
    )

    await handle(FakeMessage("запрос", chat_id=CHAT_ID), fake_bot, llm, store)

    second_request = llm.requests[1]
    roles = [m["role"] for m in second_request]
    assert roles == ["system", "user", "assistant", "tool"]
    assert "echo hi" in second_request[2]["tool_calls"][0]["arguments"]
    assert "hi" in second_request[3]["content"]
    assert "reasoning" not in second_request[2]

    assert await store.load_open_history(CHAT_ID) == [
        {"role": "user", "content": "запрос"},
        {"role": "assistant", "content": "готово"},
    ]


async def test_context_survives_bot_restart(fake_bot, tmp_path, monkeypatch):
    """Рестарт = новый store на том же файле: контекст восстанавливается,
    транскрипт инструментов прошлой обработки — нет (design D4)."""
    db_path = tmp_path / "memory.db"
    monkeypatch.setattr(main_module, "datetime", fake_datetime(T1, T2))

    first_store = MemoryStore(db_path)
    await first_store.open()
    llm = make_scripted_llm(
        [
            assistant_turn(
                content=None,
                tool_calls=[tool_call(arguments='{"command": "echo hi"}')],
                finish_reason="tool_calls",
            ),
            assistant_turn(content="первый ответ"),
        ]
    )
    await handle(
        FakeMessage("первое", chat_id=CHAT_ID), fake_bot, llm, first_store
    )
    await first_store.close()  # «рестарт» процесса бота

    second_store = MemoryStore(db_path)
    await second_store.open()
    llm.turns = [assistant_turn("второй ответ")]
    await handle(
        FakeMessage("второе", chat_id=CHAT_ID), fake_bot, llm, second_store
    )
    await second_store.close()

    assert llm.requests[2] == [
        {"role": "system", "content": SYSTEM_AT_T2},
        {"role": "user", "content": "первое"},
        {"role": "assistant", "content": "первый ответ"},
        {"role": "user", "content": "второе"},
    ]
    assert not any(
        m["role"] in ("tool",) or "tool_calls" in m
        for m in llm.requests[2]
    )


async def test_new_command_closes_session_without_llm_call(
    fake_bot, store, monkeypatch
):
    llm = make_llm_stub(reply="ответ")
    monkeypatch.setattr(main_module, "datetime", fake_datetime(T1, T2))
    await handle(
        FakeMessage("первое", chat_id=CHAT_ID), fake_bot, llm, store
    )
    llm.requests.clear()

    await handle_new(FakeMessage("/new", chat_id=CHAT_ID), fake_bot, store)

    assert fake_bot.sent[-1] == {"chat_id": CHAT_ID, "text": NEW_CHAT_CONFIRMATION}
    assert llm.requests == []  # LLM не вызывался

    llm.turns = [assistant_turn("новый ответ")]
    await handle(FakeMessage("второе", chat_id=CHAT_ID), fake_bot, llm, store)

    assert llm.requests[0] == [
        {"role": "system", "content": SYSTEM_AT_T2},
        {"role": "user", "content": "второе"},
    ]
    # Прошлая сессия осталась в хранилище и доступна поиску
    assert "первое" in await store.search_completed(CHAT_ID, "первое")


async def test_new_command_does_not_touch_other_chats(fake_bot, store):
    llm = make_llm_stub(reply="ok")
    await handle(FakeMessage("чат 42", chat_id=CHAT_ID), fake_bot, llm, store)
    await handle(
        FakeMessage("чат 43", chat_id=CHAT_ID + 1), fake_bot, llm, store
    )

    await handle_new(FakeMessage("/new", chat_id=CHAT_ID), fake_bot, store)

    assert await store.load_open_history(CHAT_ID) == []
    assert await store.load_open_history(CHAT_ID + 1) == [
        {"role": "user", "content": "чат 43"},
        {"role": "assistant", "content": "ok"},
    ]


async def test_chats_are_isolated_in_handler(fake_bot, store, monkeypatch):
    llm = make_llm_stub(reply="ok")
    monkeypatch.setattr(main_module, "datetime", fake_datetime(T1, T1))
    await handle(FakeMessage("секрет а", chat_id=1), fake_bot, llm, store)
    await handle(FakeMessage("секрет б", chat_id=2), fake_bot, llm, store)

    assert [m["role"] for m in llm.requests[1]] == ["system", "user"]
    assert llm.requests[1][1]["content"] == "секрет б"
    assert "секрет а" not in str(llm.requests[1])


async def test_steps_exhausted_message_is_sent_to_chat(fake_bot, store):
    endless = assistant_turn(
        content=None,
        tool_calls=[tool_call(arguments='{"command": "echo x"}')],
        finish_reason="tool_calls",
    )
    llm = make_scripted_llm([endless])

    await handle(FakeMessage("сложный запрос", chat_id=CHAT_ID), fake_bot, llm, store)

    assert fake_bot.sent[-1]["text"] == STEPS_EXHAUSTED_MESSAGE
    # Терминальный ответ записан как финальный ответ ассистента
    assert await store.load_open_history(CHAT_ID) == [
        {"role": "user", "content": "сложный запрос"},
        {"role": "assistant", "content": STEPS_EXHAUSTED_MESSAGE},
    ]


async def test_send_chunked_short_reply_is_one_message(fake_bot):
    text = "а" * 200

    await send_chunked(fake_bot, CHAT_ID, text)

    assert fake_bot.sent == [{"chat_id": CHAT_ID, "text": text}]


async def test_send_chunked_long_reply_is_split_in_order(fake_bot):
    text = "б" * 9000

    await send_chunked(fake_bot, CHAT_ID, text)

    lengths = [len(m["text"]) for m in fake_bot.sent]
    assert lengths == [TELEGRAM_MESSAGE_LIMIT, TELEGRAM_MESSAGE_LIMIT, 808]
    assert all(m["chat_id"] == CHAT_ID for m in fake_bot.sent)
    assert "".join(m["text"] for m in fake_bot.sent) == text


@pytest.mark.parametrize("length", [1, TELEGRAM_MESSAGE_LIMIT])
async def test_send_chunked_boundary_lengths_fit_one_message(fake_bot, length):
    text = "в" * length

    await send_chunked(fake_bot, CHAT_ID, text)

    assert len(fake_bot.sent) == 1


async def test_same_executor_resident_state_survives_messages(fake_bot, store):
    """Один executor на процесс бота: файл, созданный в первом сообщении,
    читается следующим; между сообщениями stop не вызывается."""
    executor = FakeCommandExecutor()

    llm = make_scripted_llm(
        [
            assistant_turn(
                content=None,
                tool_calls=[tool_call(arguments='{"command": "echo data > note"}')],
                finish_reason="tool_calls",
            ),
            assistant_turn(content="сохранил"),
        ]
    )
    await handle(
        FakeMessage("первое", chat_id=CHAT_ID),
        fake_bot,
        llm,
        store,
        executor=executor,
    )

    assert executor.files == {"note": "data"}
    assert executor.stop_calls == 0  # житель не удаляется между сообщениями

    llm.turns = [
        assistant_turn(
            content=None,
            tool_calls=[tool_call(arguments='{"command": "cat note"}')],
            finish_reason="tool_calls",
        ),
        assistant_turn(content="вот содержимое"),
    ]
    await handle(
        FakeMessage("второе", chat_id=CHAT_ID),
        fake_bot,
        llm,
        store,
        executor=executor,
    )

    tool_msgs = [m for m in llm.requests[3] if m["role"] == "tool"]
    assert "data" in tool_msgs[-1]["content"]  # состояние пережило сообщение
    assert executor.stop_calls == 0


async def test_main_opens_and_closes_memory_store(fake_bot, monkeypatch, tmp_path):
    """main владеет lifecycle: store открыт при старте и закрыт в finally —
    как с песочницей (design D7)."""
    import dev_helper_bot.main as main_module

    executor = FakeCommandExecutor()
    events: list[str] = []

    class FakeSession:
        async def close(self) -> None:
            events.append("session_closed")

    class FakeMainBot:
        def __init__(self, token: str, default=None) -> None:
            events.append("bot_created")
            self.session = FakeSession()

    class FakeDispatcher(main_module.Dispatcher):
        async def start_polling(self, bot) -> None:
            events.append("polling")
            memory = self["memory"]
            await memory.append_user(CHAT_ID, "внутри polling")
            assert await memory.load_open_history(CHAT_ID)
            raise RuntimeError("polling stopped")

    async def fake_prepare() -> None:
        events.append("prepared")

    monkeypatch.setattr(main_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(main_module, "telegram_token", lambda: "test-token")
    monkeypatch.setattr(main_module, "make_llm", lambda: fake_bot)
    monkeypatch.setattr(main_module, "prepare_sandbox_environment", fake_prepare)
    monkeypatch.setattr(main_module, "SandboxExecutor", lambda: executor)
    monkeypatch.setattr(main_module, "memory_db_path", lambda: str(tmp_path / "m.db"))
    monkeypatch.setattr(main_module, "Bot", FakeMainBot)
    monkeypatch.setattr(
        main_module.Dispatcher, "start_polling", FakeDispatcher.start_polling
    )

    with pytest.raises(RuntimeError, match="polling stopped"):
        await main_module.main()

    assert events == ["bot_created", "prepared", "polling", "session_closed"]
    assert executor.stop_calls == 1  # best-effort stop при завершении бота
    # Store закрыт: переоткрытие того же файла видит записанное
    reopened = MemoryStore(tmp_path / "m.db")
    await reopened.open()
    try:
        assert await reopened.load_open_history(CHAT_ID) == [
            {"role": "user", "content": "внутри polling"}
        ]
    finally:
        await reopened.close()
