# Implementation overview

Cancel Capture is a single Python package with three application services and adapter-based
external boundaries. [`build_services`](../src/cancel_capture/container.py) is the composition
root: it creates the content-addressed asset store, SQLite catalog, Pillow image processor,
metadata extractor, and separate vision, text, and embedding providers resolved through the
[`provider_registry`](../src/cancel_capture/provider_registry.py), then injects them into the
ingestion, review, and search services. `build_narrative_services` composes the Streamlit-only
narrative + clustering stack on top of that graph, and is not required by the bot or CLI.

The application layer depends on the typed
[`Protocol` ports](../src/cancel_capture/ports.py), not on Telegram, OpenAI, Pillow, or SQLite.
Immutable dataclasses in [`models.py`](../src/cancel_capture/models.py) and
[`narrative_models.py`](../src/cancel_capture/narrative_models.py) carry validated bounding
boxes, observations, descriptions, provider identities, assets, review candidates, Telegram
provenance, search results, and narrative artifacts between those boundaries. The shipped
composition uses OpenAI-compatible APIs and SQLite, but those choices remain outside the
application services.

## Entry points

- The [`cancel-capture` CLI](../src/cancel_capture/cli.py) initializes and diagnoses storage,
  runs the bot, imports channel history, and performs terminal search.
- The owner-only [Telegram bot](../src/cancel_capture/adapters/telegram_bot.py) accepts original
  image documents, creates private review cards, and publishes approved crops.
- The [Telethon importer](../src/cancel_capture/adapters/telethon_history.py) backfills existing
  channel images without reposting them.
- The private [Streamlit application](../src/cancel_capture/streamlitapp/app.py) supports upload,
  reanalysis, catalog inspection, local rejection, bilingual search, hierarchical clustering,
  and narrative generation. It deliberately does not publish to the channel.

## Deep-dive pages

- [Ingestion pipeline](ingestion.md) — how a raw upload becomes crops, descriptions, embeddings,
  and catalog rows in a single transaction.
- [Review and publication](review.md) — the sign-level state machine, conditional publish
  claims, and Telegram delivery boundary.
- [Persistence, retrieval, and verification](persistence.md) — SQLite catalog, search, and how
  `./check.sh` proves the whole thing.
- [Streamlit-only experiments](experiments.md) — clustering, narrative generation, progress
  reporting, and provider switching.

Operational setup, backups, and failure recovery live in the [runbook](runbook.md); design
rationale and privacy boundaries in [architecture](architecture.md).
