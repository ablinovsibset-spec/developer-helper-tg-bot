# Docker-интеграционные тесты песочницы (маркер `docker`, запуск: pytest -m docker).
#
# Проверяют реальные сценарии спецификации docker-sandbox против живого
# Docker-демона: долгоживущий контейнер-жилец на процесс бота (персистентное
# файловое состояние и пакеты между сообщениями и /new), удаление при
# завершении бота, sweep осиротевших песочниц, таймаут внутри контейнера,
# самовосстановление и сборку образа.
from __future__ import annotations

import asyncio
import uuid

import pytest

from dev_helper_bot.agent import MAX_LLM_STEPS, STEPS_EXHAUSTED_MESSAGE, run_agent
from dev_helper_bot.sandbox import SANDBOX_IMAGE, SandboxExecutor, ensure_image, sweep_orphaned_sandboxes
from dev_helper_bot.tools import EXEC_TOOL_SPEC

from tests.conftest import assistant_turn, make_scripted_llm, tool_call

pytestmark = pytest.mark.docker


@pytest.fixture(scope="module")
def sandbox_image() -> str:
    asyncio.run(ensure_image())
    return SANDBOX_IMAGE


@pytest.fixture(autouse=True)
def _require_image(sandbox_image: str) -> None:
    """Образ собран до начала любого docker-теста."""


def _tool_turn(command: str) -> dict:
    return assistant_turn(
        content=None,
        tool_calls=[tool_call(arguments='{"command": "%s"}' % command)],
        finish_reason="tool_calls",
    )


async def _run_agent_with(llm, history, executor) -> str:
    return await run_agent(llm, history, tools=[EXEC_TOOL_SPEC], executor=executor)


async def _docker_cli(*args: str, check: bool = True) -> str:
    process = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await process.communicate()
    if check and process.returncode != 0:
        raise AssertionError(f"docker {args} failed: {process.returncode}")
    return stdout.decode().strip()


async def _container_exists(container_id: str | None) -> bool:
    if not container_id:
        return False
    process = await asyncio.create_subprocess_exec(
        "docker", "container", "inspect", container_id
    )
    return await process.wait() == 0


async def test_exec_runs_as_unprivileged_user_in_workdir():
    executor = SandboxExecutor()
    try:
        result = await executor.execute('echo "uid=$(id -u) home=$HOME pwd=$(pwd)"')

        assert result.exit_code == 0
        assert "uid=1000" in result.stdout
        assert "home=/work" in result.stdout
        assert "pwd=/work" in result.stdout
    finally:
        await executor.stop()


async def test_resident_not_removed_between_messages_removed_on_stop():
    """4.3: контейнер-житель один на все сообщения, удаляется при завершении
    бота (stop), а не после каждого агентного цикла."""
    executor = SandboxExecutor()
    first_llm = make_scripted_llm(
        [_tool_turn("echo step-one"), assistant_turn(content="готово 1")]
    )
    try:
        await _run_agent_with(
            first_llm, [{"role": "user", "content": "первое"}], executor
        )
        first_id = executor.container_id
        assert first_id is not None  # создан лениво при первом exec

        second_llm = make_scripted_llm(
            [_tool_turn("echo step-two"), assistant_turn(content="готово 2")]
        )
        await _run_agent_with(
            second_llm, [{"role": "user", "content": "второе"}], executor
        )

        assert executor.container_id == first_id  # тот же житель
        assert await _container_exists(first_id) is True
    finally:
        await executor.stop()

    assert await _container_exists(first_id) is False  # убран при завершении


async def test_resident_not_removed_after_steps_exhausted():
    llm = make_scripted_llm([_tool_turn("echo again")])
    history = [{"role": "user", "content": "сделай"}]
    executor = SandboxExecutor()

    try:
        reply = await _run_agent_with(llm, history, executor)

        assert reply == STEPS_EXHAUSTED_MESSAGE
        assert len(llm.requests) == MAX_LLM_STEPS
        assert await _container_exists(executor.container_id) is True
    finally:
        await executor.stop()


async def test_file_survives_steps_within_one_message():
    llm = make_scripted_llm(
        [
            _tool_turn("echo payload > note"),
            _tool_turn("cat note"),
            assistant_turn(content="итог"),
        ]
    )
    history = [{"role": "user", "content": "сделай"}]
    executor = SandboxExecutor()

    try:
        reply = await _run_agent_with(llm, history, executor)

        assert reply == "итог"
        cat_result = llm.requests[2][4]["content"]
        assert "payload" in cat_result
        assert "exit_code: 0" in cat_result
    finally:
        await executor.stop()


async def test_file_created_in_one_message_is_readable_in_next():
    """4.1: файл переживает сообщения; /new (новая пустая история LLM)
    не затрагивает файловое состояние жителя."""
    executor = SandboxExecutor()
    try:
        first_llm = make_scripted_llm(
            [_tool_turn("echo payload > note"), assistant_turn(content="сохранил")]
        )
        await _run_agent_with(
            first_llm, [{"role": "user", "content": "сохрани файл"}], executor
        )

        # «/new»: контекст LLM сброшен — новая история, тот же житель.
        second_llm = make_scripted_llm(
            [_tool_turn("cat note"), assistant_turn(content="прочитал")]
        )
        reply = await _run_agent_with(
            second_llm, [{"role": "user", "content": "прочитай файл"}], executor
        )

        assert reply == "прочитал"
        cat_result = second_llm.requests[1][2]["content"]
        assert "payload" in cat_result
        assert "exit_code: 0" in cat_result
    finally:
        await executor.stop()


async def test_user_pip_package_survives_messages():
    """4.2: пакет из `pip install --user` доступен в последующих сообщениях
    без переустановки."""
    executor = SandboxExecutor()
    try:
        install = await executor.execute(
            "pip install --user six >/dev/null 2>&1 && echo installed"
        )
        assert install.exit_code == 0
        assert "installed" in install.stdout

        check = await executor.execute(
            "python3 -c \"import six; print('six', six.__version__)\""
        )

        assert check.exit_code == 0
        assert check.stdout.startswith("six ")
    finally:
        await executor.stop()


async def test_timeout_inside_container_reports_timed_out():
    executor = SandboxExecutor()
    try:
        result = await executor.execute("sleep 5", timeout=1.0)

        assert result.timed_out is True
        assert result.exit_code in (124, 143)
        # Житель переживает таймаут команды: контейнер на месте.
        assert await _container_exists(executor.container_id) is True
    finally:
        await executor.stop()


async def test_self_heal_after_resident_death_clean_state_and_retry():
    """4.4: смерть жителя посреди работы — пересоздание с чистым состоянием,
    повтор команды, бот не падает."""
    executor = SandboxExecutor()
    try:
        await executor.execute("echo payload > note")
        old_id = executor.container_id
        assert old_id is not None

        # Внешнее убийство жителя между сообщениями.
        await _docker_cli("rm", "-f", old_id)

        result = await executor.execute("echo alive")

        assert result.exit_code == 0
        assert "alive" in result.stdout  # команда выполнена повторно

        note = await executor.execute("cat note")

        assert note.exit_code != 0  # состояние чистое: файла нет
        assert "payload" not in note.stdout
        assert executor.container_id != old_id
        assert await _container_exists(executor.container_id) is True
    finally:
        await executor.stop()


async def test_sweep_removes_orphaned_labeled_containers():
    # Сирота после «аварийного завершения бота»: executor без stop().
    crashed = SandboxExecutor()
    await crashed.execute("echo orphan-marker")
    orphan_ids = [crashed.container_id]
    assert orphan_ids[0] is not None

    # Отдельный посторонний контейнер с меткой бота.
    orphan_ids.append(
        await _docker_cli(
            "run",
            "-d",
            "--label",
            "dev-helper-bot.sandbox=true",
            "alpine:3.20",
            "sleep",
            "infinity",
        )
    )

    swept = await sweep_orphaned_sandboxes()

    assert swept >= 2
    for orphan_id in orphan_ids:
        assert await _container_exists(orphan_id) is False


async def test_ensure_image_builds_missing_image():
    tag = f"dev-helper-bot-sandbox-test:{uuid.uuid4().hex[:12]}"

    try:
        await ensure_image(image=tag)

        process = await asyncio.create_subprocess_exec(
            "docker", "image", "inspect", tag
        )
        assert await process.wait() == 0
    finally:
        await _docker_cli("rmi", "-f", tag, check=False)
