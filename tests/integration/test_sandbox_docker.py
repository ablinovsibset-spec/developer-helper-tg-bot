# Docker-интеграционные тесты песочницы (маркер `docker`, запуск: pytest -m docker).
#
# Проверяют реальные сценарии спецификации docker-sandbox против живого
# Docker-демона: жизненный цикл контейнера на агентный цикл, файловое
# состояние в пределах сообщения, таймаут внутри контейнера,
# самовосстановление, sweep осиротевших песочниц и сборку образа.
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
        await executor.start()
        result = await executor.execute('echo "uid=$(id -u) home=$HOME pwd=$(pwd)"')

        assert result.exit_code == 0
        assert "uid=1000" in result.stdout
        assert "home=/work" in result.stdout
        assert "pwd=/work" in result.stdout
    finally:
        await executor.stop()


async def test_container_removed_after_agent_cycle():
    llm = make_scripted_llm(
        [_tool_turn("echo step"), assistant_turn(content="готово")]
    )
    history = [{"role": "user", "content": "сделай"}]
    seen: list[str | None] = []

    class TrackingExecutor(SandboxExecutor):
        async def execute(self, command, timeout=30.0):
            seen.append(self.container_id)
            return await super().execute(command, timeout)

    reply = await _run_agent_with(llm, history, TrackingExecutor())

    assert reply == "готово"
    assert seen and await _container_exists(seen[0]) is False


async def test_container_removed_after_steps_exhausted():
    llm = make_scripted_llm([_tool_turn("echo again")])
    history = [{"role": "user", "content": "сделай"}]
    executor = SandboxExecutor()

    reply = await _run_agent_with(llm, history, executor)

    assert reply == STEPS_EXHAUSTED_MESSAGE
    assert len(llm.requests) == MAX_LLM_STEPS
    assert await _container_exists(executor.container_id) is False


async def test_file_survives_steps_within_one_message():
    llm = make_scripted_llm(
        [
            _tool_turn("echo payload > note"),
            _tool_turn("cat note"),
            assistant_turn(content="итог"),
        ]
    )
    history = [{"role": "user", "content": "сделай"}]

    reply = await _run_agent_with(llm, history, SandboxExecutor())

    assert reply == "итог"
    cat_result = llm.requests[2][4]["content"]
    assert "payload" in cat_result
    assert "exit_code: 0" in cat_result


async def test_state_does_not_survive_between_messages():
    first = SandboxExecutor()
    try:
        await first.start()
        await first.execute("echo payload > note")
        first_id = first.container_id
    finally:
        await first.stop()

    assert await _container_exists(first_id) is False

    second = SandboxExecutor()
    try:
        await second.start()
        result = await second.execute("cat note")
    finally:
        await second.stop()

    assert result.exit_code != 0
    assert "No such file" in result.stderr


async def test_timeout_inside_container_reports_timed_out():
    executor = SandboxExecutor()
    try:
        await executor.start()

        result = await executor.execute("sleep 5", timeout=1.0)

        assert result.timed_out is True
        assert result.exit_code in (124, 143)
    finally:
        await executor.stop()


async def test_self_heal_after_container_death():
    executor = SandboxExecutor()
    try:
        await executor.start()

        # Внешнее убийство контейнера посреди «цикла».
        await _docker_cli("rm", "-f", executor.container_id)

        result = await executor.execute("echo alive")

        assert result.exit_code == 0
        assert "alive" in result.stdout
        assert await _container_exists(executor.container_id) is True
    finally:
        await executor.stop()


async def test_sweep_removes_orphaned_labeled_containers():
    orphan_id = await _docker_cli(
        "run",
        "-d",
        "--label",
        "dev-helper-bot.sandbox=true",
        "alpine:3.20",
        "sleep",
        "infinity",
    )

    swept = await sweep_orphaned_sandboxes()

    assert swept >= 1
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
