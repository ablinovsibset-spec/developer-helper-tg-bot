# AGENTS.md

Guidance for Cursor / OpenCode coding agents working in this repository.
Not for the Telegram runtime agent (that uses `skills/` + its system prompt).

## Project

Telegram bot that runs a local LLM agent loop inside Docker Sandboxes (`sbx`).
Installable package: `src/dev_helper_bot/`.

## Verify

| Change | Required check |
| --- | --- |
| `src/`, `tests/`, `pyproject.toml`, deps | `pytest` (hermetic default; docker tests excluded) |
| Docs / `AGENTS.md` / skill markdown only | Tests not required |
| `sandbox.py`, `Dockerfile`, `scripts/sbx-*.sh` | `pytest`, and `pytest -m docker` if Docker is available; otherwise state the gap |
| Dependency changes | `pytest`, and remind to re-run `scripts/sbx-setup.sh` (venv lives on the sandbox VM disk) |

There is no lint / typecheck / format tooling in this repo — do not invent a pipeline.

## Gotchas

- Keep `MEMORY_DB_PATH` and `OBS_DB_PATH` on **VM-local disk** (defaults under `~/.local/share/…`). Do not put SQLite files on the workspace mount — locks/WAL break.
- Workspace `.env` (Telegram token) is a **trust boundary**: available to the bot process, not mounted into the exec resident container.
- `scripts/sbx-start.sh` sets `LLM_BASE_URL` (and `OBS_WEB_HOST=0.0.0.0`) via env; that wins over `.env` `localhost` (dotenv does not overwrite existing env).
- Use an **editable** install (`pip install -e .`). `skills/` and `Dockerfile` resolve via `Path(__file__).parents[2]`.
- If the observability web port is busy, the bot fails fast and does not start polling.
- Sandbox keepalive: without the keepalive session from `sbx-start.sh`, the sbx VM stops ~30s after the last exec session.

## Modules

`main`, `agent`, `tools`, `sandbox`, `memory`, `telemetry`, `obs_web` / `obs_web_pages`, `skills`, `config`, `llm` — see [docs/architecture.md](docs/architecture.md).

## OpenSpec

**Required** when observable behavior changes, an existing `openspec/specs/*` becomes false, or you refactor public modules / rename tools.

**Skip** for typos, docs-only edits, fixing a test to match an already-correct spec, or DX noise that does not change the runtime contract.

Checklist:

1. Trigger matches above → create an OpenSpec change
2. Update delta specs
3. Implement
4. Verify (see Verify)
5. Archive the change

Follow the OpenSpec propose / apply / archive skills for artifact details.

## Conventions

- English identifiers; Russian comments and human-facing artifacts are OK
- Secrets only in `.env` — never commit them
- Keep `requirements.txt` in sync with `pyproject.toml` dependencies
- Prefer injectable test seams (`LLMClient`, `CommandExecutor`, fake LLM) over hard-wiring
- Prefer `openspec/specs/`, README, and code over `openspec/config.yaml` `context:` — that block may be stale

## Pointers

| Need | Where |
| --- | --- |
| Ops / runbook / env tables | [README.md](README.md) |
| Behavioral truth | [openspec/specs/](openspec/specs/) |
| Module map, message flow, test seams | [docs/architecture.md](docs/architecture.md) |
| Runtime skill bodies | [skills/](skills/) |
