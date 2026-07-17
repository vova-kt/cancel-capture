# Repository guidance

## Purpose and invariants

- Preserve every uploaded original byte-for-byte before decoding or calling an external API.
- Treat camera metadata, Telegram sessions, credentials, originals, and the SQLite database as
  private runtime data. Never commit or log them.
- Public channel posts use metadata-stripped crops. Raw metadata and GPS never go into captions.
- Telegram callbacks and history imports are replayable. State transitions and source imports must
  remain conditional and idempotent.
- Keep vision, text, embedding, storage, catalog, and publishing behind `Protocol` interfaces. Do
  not couple application services to OpenAI, Telegram, Streamlit, or SQLite implementations.

## Commands

- Install locked development dependencies: `uv sync --extra dev`
- Run every required local check: `./check.sh`
- Initialize or inspect configuration: `uv run cancel-capture doctor`
- Run the bot: `uv run cancel-capture bot`
- Import channel history: `uv run cancel-capture import-history`
- Run the development UI: `uv run streamlit run src/cancel_capture/streamlit_app.py`
- Run in Docker: `docker compose up -d bot`

## Engineering rules

1. Update the relevant concise `docs/*.md` page and this file in the same change whenever
   behavior, architecture, contracts, commands, or completion criteria change.
2. Fix root causes. Do not suppress exceptions, special-case a failing fixture, silently degrade,
   or ship a workaround unless the underlying constraint and tradeoff are documented explicitly.
3. `src/` is strict Pyright territory. Avoid `Any`; annotate new public functions and classes and
   keep boundary conversions explicit.
4. Pluggable interfaces are `Protocol`s, not inheritance hierarchies. User-facing configuration is
   immutable and uses `@dataclass(frozen=True)`.
5. New behavior requires tests. Cover failure, retry, and idempotency paths for external boundaries.
   Tests and CI never contact live Telegram or model APIs.
6. SQLite changes use forward-only, transactional migrations. Enable foreign keys, WAL, and a busy
   timeout on every connection. Store relative volume paths, never host-specific absolute paths.
7. Keep the exact original. Apply detected boxes to its oriented full-resolution pixels. Derived
   analysis images and crops are stripped of metadata; never overwrite an asset in place. Create a
   disposable Bot-API-safe rendition for Telegram delivery without changing the archival crop.
8. Treat Telegram identifiers, callback payloads, filenames, MIME types, model output, EXIF, and
   imported captions as untrusted input. Authorize with the configured numeric owner ID.
9. A publish attempt moves through a conditional review state. Never auto-retry an ambiguous
   Telegram send that could have succeeded and created a duplicate channel post.
10. Documentation explains rationale, tradeoffs, invariants, and operations. It does not duplicate
    signatures, exports, schema field lists, or file trees; link to source instead.
11. Comments and docstrings justify non-obvious constraints. Do not narrate code, add banner module
    docstrings, or repeat information visible in names and types.
12. Preserve unrelated user changes. Do not commit generated data, `.env`, Telegram session files,
    databases, uploaded images, analysis assets, or provider responses.

## Definition of done

- The requested behavior and its unhappy paths have focused tests.
- `./check.sh` passes unchanged, including Ruff formatting/lint, strict Pyright, and pytest.
- Relevant operational or architectural documentation is updated without duplicating code.
- No secret, private metadata, runtime asset, or session credential appears in the diff or logs.
- External side effects are reported accurately; unverified live-provider behavior is not claimed.

This file follows current [OpenAI AGENTS.md guidance](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
and [Anthropic CLAUDE.md guidance](https://code.claude.com/docs/en/best-practices): keep durable
instructions short, concrete, repository-specific, and paired with executable verification.
