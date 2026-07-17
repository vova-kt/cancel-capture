# Persistence, retrieval, and verification

## SQLite catalog

[`SQLiteCatalog`](../src/cancel_capture/adapters/sqlite_catalog.py) owns schema migration and all
catalog state. Every connection enables foreign keys, WAL, and a busy timeout; migrations and
multi-record ingestion are forward-only transactions. The catalog stores:

- content-addressed asset records and photo/sign relationships;
- normalized metadata alongside namespaced raw metadata JSON;
- raw and refined detection boxes, confidence, visible text, and topics;
- bilingual descriptions and provider identities;
- Telegram inbound, history, preview, and channel-post provenance;
- review tokens, state, errors, and append-only review events;
- history-import failure attempts;
- FTS5 search documents plus little-endian float32 embedding blobs; and
- per-sign visual embeddings on a separate table (introduced by migration 7) so text and image
  vectors can evolve independently.

## Search

[`SearchService`](../src/cancel_capture/application/search.py) embeds the bilingual query, selects
only stored vectors with the same provider, namespace, model, and dimensions, and calculates
exact cosine similarity in process. An FTS5 match adds a small fixed lexical bonus. Results may
be scoped to photos or signs and include their review status and asset path.

## Configuration, deployment, and verification

[`AppConfig`](../src/cancel_capture/config.py) loads immutable storage, provider, bot, and
history settings from environment variables. Vision, text, embedding, and narrative roles are
independently configurable. A custom endpoint receives a derived non-secret identity namespace
unless an explicit namespace is supplied, preventing vectors from distinct deployments from being
mixed accidentally.

The package supports direct `uv` execution and the committed Docker/Compose setup. Runtime
assets, the SQLite database, and the Telethon session live in the configured private data
volume. Operational setup, backups, and failure recovery are covered by the
[runbook](runbook.md); design rationale and privacy boundaries are covered by
[architecture](architecture.md).

The committed tests use fakes for external boundaries and cover asset handling, image
processing, metadata, structured provider adapters, ingestion idempotency, search, review
conflicts and recovery, Telegram behavior, history replay, configuration, clustering, narrative
selection, narrative persistence, prompt rendering, progress reporting, provider registry
dispatch, and Streamlit startup. The repository-wide local verification command is
`./check.sh`, which runs formatting/lint, strict Pyright, and pytest without live provider
calls.

## Related pages

- [Implementation overview](implementation.md) — index and runtime shape.
- [Ingestion pipeline](ingestion.md) — what writes rows into the catalog.
- [Review and publication](review.md) — state transitions guarded by conditional catalog updates.
- [Streamlit-only experiments](experiments.md) — how clustering and narrative reuse the catalog.
