# Architecture

Map of `src/dev_helper_bot/` for coding agents. Behavioral contracts and limits live in `openspec/specs/` and the code — this file is only a layout guide.

## Modules

| Module | Responsibility |
| --- | --- |
| `main` | Telegram entry: polling, `/new`, owns executor / memory / telemetry / obs web lifecycle |
| `agent` | Agent loop: LLM turns, tool dispatch, validation, tool-output compaction |
| `tools` | Tool specs and handlers: `exec`, `read_file`, `get_skill`, `search_history`, `list_sessions`; `CommandExecutor` / `HistorySearcher` protocols |
| `sandbox` | Long-lived Docker resident for `exec` (`SandboxExecutor`) |
| `memory` | SQLite chat sessions, restore open history, search |
| `telemetry` | Observability SQLite store, `RunRecorder`, `ObservingClient` wrapper |
| `obs_web` / `obs_web_pages` | In-process aiohttp dashboard + HTML |
| `skills` | Load markdown skills (YAML frontmatter), system catalog, `build_request_messages` |
| `config` | Env settings, `make_llm()`, path helpers |
| `llm` | `LLMClient` protocol (`base`) and OpenAI-compatible client (`openai_compat`) |

## Message flow

1. Telegram text → `main.handle_text`
2. Load open session history from `memory`; build messages via `skills.build_request_messages` (system catalog + history + current user text)
3. `agent.run_agent` calls the LLM (optionally wrapped by `ObservingClient`)
4. Tool calls go through `tools` → `sandbox` (`exec`), filesystem (`read_file`), skill bodies (`get_skill`), or memory search/list
5. Final assistant text is stored in memory and sent back to Telegram (chunked)

## Test seams

Inject these instead of production wiring:

| Seam | Production | Tests |
| --- | --- | --- |
| `LLMClient` | `config.make_llm()` → `OpenAICompatibleClient` | Fake / stub client |
| `CommandExecutor` | `SandboxExecutor` | In-memory or fake executor |
| `MemoryStore` / history search | SQLite paths from config | Temp DB or fakes |
| `TelemetryStore` | Optional; best-effort | Omit or temp DB |

Hermetic suite: `pytest` (default excludes `@pytest.mark.docker`). Docker-marked tests need a real daemon.
