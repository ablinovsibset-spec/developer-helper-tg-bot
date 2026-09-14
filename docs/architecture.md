# Architecture

Map of `src/dev_helper_bot/` for coding agents. Behavioral contracts and limits live in `openspec/specs/` and the code — this file is only a layout guide.

## Modules

| Module | Responsibility |
| --- | --- |
| `main` | Telegram entry: polling, `/new`, `/documents`, `/delete`, document upload; owns executor / memory / telemetry / document store / obs web lifecycle |
| `agent` | Agent loop: LLM turns, tool dispatch, validation, tool-output compaction |
| `tools` | Tool specs and handlers: `exec`, `read_file`, `get_skill`, `search_history`, `list_sessions`, `search_documents`; `CommandExecutor` / `HistorySearcher` / `DocumentSearcher` protocols |
| `sandbox` | Long-lived Docker resident for `exec` (`SandboxExecutor`) |
| `memory` | SQLite chat sessions, restore open history, search |
| `documents` | Text extraction (txt/md/docx/pdf), chunking, indexing limits — pure functions |
| `document_store` | SQLite + sqlite-vec index (`sqlean.py` driver), KNN partitioned by owner, `UserDocumentSearcher` |
| `embeddings` | `EmbeddingClient` protocol (`base`) and OpenAI-compatible `/embeddings` client |
| `telemetry` | Observability SQLite store, `RunRecorder`, `ObservingClient` wrapper |
| `obs_web` / `obs_web_pages` | In-process aiohttp dashboard + HTML |
| `skills` | Load markdown skills (YAML frontmatter), system catalog, `build_request_messages` |
| `config` | Env settings, `make_llm()`, `make_embeddings()`, path helpers |
| `llm` | `LLMClient` protocol (`base`) and OpenAI-compatible client (`openai_compat`) |

## Message flow

1. Telegram text → `main.handle_text`
2. Load open session history from `memory`; build messages via `skills.build_request_messages` (system catalog + history + current user text)
3. `agent.run_agent` calls the LLM (optionally wrapped by `ObservingClient`)
4. Tool calls go through `tools` → `sandbox` (`exec`), filesystem (`read_file`), skill bodies (`get_skill`), memory search/list, or the document index (`search_documents`)
5. Final assistant text is stored in memory and sent back to Telegram (chunked)

Documents take a separate path: `main.handle_document` validates, downloads, extracts,
chunks, embeds and writes the index without touching the agent loop or memory.

## Test seams

Inject these instead of production wiring:

| Seam | Production | Tests |
| --- | --- | --- |
| `LLMClient` | `config.make_llm()` → `OpenAICompatibleClient` | Fake / stub client |
| `EmbeddingClient` | `config.make_embeddings()` → `OpenAICompatibleEmbeddingClient` | `FakeEmbeddingClient` (deterministic hashed bag-of-words) |
| `CommandExecutor` | `SandboxExecutor` | In-memory or fake executor |
| `MemoryStore` / history search | SQLite paths from config | Temp DB or fakes |
| `DocumentSearcher` | `UserDocumentSearcher` over `DocumentStore` | Temp-path store or fake searcher |
| `TelemetryStore` | Optional; best-effort | Omit or temp DB |

Hermetic suite: `pytest` (default excludes `@pytest.mark.docker`). Docker-marked tests need a real daemon.
