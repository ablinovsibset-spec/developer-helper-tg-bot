from __future__ import annotations

from datetime import datetime

import pytest

from dev_helper_bot import main as main_module
from dev_helper_bot.agent import STEPS_EXHAUSTED_MESSAGE
from dev_helper_bot.llm import LLMUnavailable
from dev_helper_bot.main import (
    NEW_CHAT_CONFIRMATION,
    TELEGRAM_MESSAGE_LIMIT,
    WAITING_MESSAGE,
    handle_new,
    handle_text,
    send_chunked,
)
from dev_helper_bot.skills import SANDBOX_ENV_LINE
from dev_helper_bot.tools import EXEC_TOOL_SPEC

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


async def handle(
    message,
    fake_bot,
    llm,
    histories=None,
    skills=SKILLS,
    executor=None,
):
    histories = histories if histories is not None else {}
    executor = executor if executor is not None else FakeCommandExecutor()
    await handle_text(message, fake_bot, llm, histories, skills, executor)
    return histories


async def test_handle_text_sends_waiting_then_final_reply(fake_bot):
    llm = make_llm_stub(reply="ответ модели")

    await handle(FakeMessage("привет", chat_id=CHAT_ID), fake_bot, llm)

    assert fake_bot.sent == [
        {"chat_id": CHAT_ID, "text": WAITING_MESSAGE},
        {"chat_id": CHAT_ID, "text": "ответ модели"},
    ]


async def test_handle_text_prompt_goes_to_llm_with_system_and_tools(
    fake_bot, monkeypatch
):
    llm = make_llm_stub(reply="ok")
    monkeypatch.setattr(main_module, "datetime", fake_datetime(T1))

    await handle(FakeMessage("привет", chat_id=CHAT_ID), fake_bot, llm)

    assert llm.requests[0] == [
        {"role": "system", "content": SYSTEM_AT_T1},
        {"role": "user", "content": "привет"},
    ]
    assert llm.tools_per_request == [[EXEC_TOOL_SPEC]]


async def test_handle_text_llm_unavailable_sends_friendly_error(fake_bot):
    llm = make_llm_stub(error=LLMUnavailable("connection refused"))

    await handle(FakeMessage("привет", chat_id=CHAT_ID), fake_bot, llm)

    assert len(fake_bot.sent) == 2  # ⏳ и сообщение об ошибке
    assert fake_bot.sent[-1]["chat_id"] == CHAT_ID
    assert "недоступна" in fake_bot.sent[-1]["text"]


async def test_context_is_kept_between_messages(fake_bot, monkeypatch):
    llm = make_llm_stub(reply="первый ответ")
    monkeypatch.setattr(main_module, "datetime", fake_datetime(T1, T2))
    histories = await handle(
        FakeMessage("первое", chat_id=CHAT_ID), fake_bot, llm
    )

    llm.turns = [assistant_turn("второй ответ")]
    await handle(FakeMessage("второе", chat_id=CHAT_ID), fake_bot, llm, histories)

    assert llm.requests[1] == [
        {"role": "system", "content": SYSTEM_AT_T2},
        {"role": "user", "content": "первое"},
        {"role": "assistant", "content": "первый ответ"},
        {"role": "user", "content": "второе"},
    ]


async def test_system_message_is_refreshed_between_messages(fake_bot, monkeypatch):
    llm = make_llm_stub(reply="первый ответ")
    monkeypatch.setattr(main_module, "datetime", fake_datetime(T1, T2))
    histories = await handle(
        FakeMessage("первое", chat_id=CHAT_ID), fake_bot, llm
    )

    assert histories[CHAT_ID][0] == {"role": "system", "content": SYSTEM_AT_T1}

    llm.turns = [assistant_turn("второй ответ")]
    await handle(FakeMessage("второе", chat_id=CHAT_ID), fake_bot, llm, histories)

    assert histories[CHAT_ID][0] == {"role": "system", "content": SYSTEM_AT_T2}
    assert [m["role"] for m in histories[CHAT_ID]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert histories[CHAT_ID][1:] == [
        {"role": "user", "content": "первое"},
        {"role": "assistant", "content": "первый ответ"},
        {"role": "user", "content": "второе"},
        {"role": "assistant", "content": "второй ответ"},
    ]


async def test_tool_transcript_is_kept_in_history(fake_bot):
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

    histories = await handle(
        FakeMessage("запрос", chat_id=CHAT_ID), fake_bot, llm
    )

    second_request = llm.requests[1]
    roles = [m["role"] for m in second_request]
    assert roles == ["system", "user", "assistant", "tool"]
    assert "echo hi" in second_request[2]["tool_calls"][0]["arguments"]
    assert "hi" in second_request[3]["content"]
    assert "reasoning" not in histories[CHAT_ID][2]
    assert histories[CHAT_ID][-1] == {"role": "assistant", "content": "готово"}


async def test_steps_exhausted_message_is_sent_to_chat(fake_bot):
    endless = assistant_turn(
        content=None,
        tool_calls=[tool_call(arguments='{"command": "echo x"}')],
        finish_reason="tool_calls",
    )
    llm = make_scripted_llm([endless])

    await handle(FakeMessage("сложный запрос", chat_id=CHAT_ID), fake_bot, llm)

    assert fake_bot.sent[-1]["text"] == STEPS_EXHAUSTED_MESSAGE


async def test_new_command_resets_context_without_llm_call(fake_bot, monkeypatch):
    llm = make_llm_stub(reply="ответ")
    monkeypatch.setattr(main_module, "datetime", fake_datetime(T1, T2))
    histories = await handle(
        FakeMessage("первое", chat_id=CHAT_ID), fake_bot, llm
    )
    llm.requests.clear()

    await handle_new(FakeMessage("/new", chat_id=CHAT_ID), fake_bot, histories)

    assert fake_bot.sent[-1] == {"chat_id": CHAT_ID, "text": NEW_CHAT_CONFIRMATION}
    assert llm.requests == []  # LLM не вызывался

    llm.turns = [assistant_turn("новый ответ")]
    await handle(FakeMessage("второе", chat_id=CHAT_ID), fake_bot, llm, histories)

    assert llm.requests[0] == [
        {"role": "system", "content": SYSTEM_AT_T2},
        {"role": "user", "content": "второе"},
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


async def test_same_executor_resident_state_survives_messages(fake_bot):
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
    histories = await handle(
        FakeMessage("первое", chat_id=CHAT_ID),
        fake_bot,
        llm,
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
        histories=histories,
        executor=executor,
    )

    tool_msg = llm.requests[3][7]  # второй вызов LLM второго сообщения: tool-сообщение
    assert tool_msg["role"] == "tool"
    assert "data" in tool_msg["content"]  # состояние пережило сообщение
    assert executor.stop_calls == 0


async def test_main_creates_single_executor_and_stops_it_on_shutdown(
    fake_bot, monkeypatch
):
    """main владеет lifecycle: один executor на процесс, stop() в finally
    срабатывает и при ошибке polling, заодно закрывается session бота."""
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
            dp_executor = self["executor"]
            assert dp_executor is executor
            raise RuntimeError("polling stopped")

    async def fake_prepare() -> None:
        events.append("prepared")

    monkeypatch.setattr(main_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(main_module, "telegram_token", lambda: "test-token")
    monkeypatch.setattr(main_module, "make_llm", lambda: fake_bot)
    monkeypatch.setattr(main_module, "prepare_sandbox_environment", fake_prepare)
    monkeypatch.setattr(main_module, "SandboxExecutor", lambda: executor)
    monkeypatch.setattr(main_module, "Bot", FakeMainBot)
    monkeypatch.setattr(
        main_module.Dispatcher, "start_polling", FakeDispatcher.start_polling
    )

    with pytest.raises(RuntimeError, match="polling stopped"):
        await main_module.main()

    assert events == ["bot_created", "prepared", "polling", "session_closed"]
    assert executor.stop_calls == 1  # best-effort stop при завершении бота
