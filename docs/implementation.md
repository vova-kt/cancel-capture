# Implementation overview

This page describes the committed implementation at `411dfa4` (`Update GitHub Actions runtimes`).
It intentionally excludes uncommitted working-tree changes.

## Runtime shape

Cancel Capture is a single Python package with three application services and adapter-based
external boundaries. [`build_services`](../src/cancel_capture/container.py) is the composition root:
it creates the content-addressed asset store, SQLite catalog, Pillow image processor, metadata
extractor, and separate OpenAI vision, text, and embedding adapters, then injects them into
ingestion, review, and search services.

The application layer depends on the typed
[`Protocol` ports](../src/cancel_capture/ports.py), not on Telegram, OpenAI, Pillow, or SQLite.
Immutable dataclasses in
[`models.py`](../src/cancel_capture/models.py) carry validated bounding boxes, observations,
descriptions, provider identities, assets, review candidates, Telegram provenance, and search
results between those boundaries. The shipped composition uses OpenAI-compatible APIs and SQLite,
but those choices remain outside the application services.

There are four user-facing entry points:

- The [`cancel-capture` CLI](../src/cancel_capture/cli.py) initializes and diagnoses storage, runs
  the bot, imports channel history, and performs terminal search.
- The owner-only [Telegram bot](../src/cancel_capture/adapters/telegram_bot.py) accepts original
  image documents, creates private review cards, and publishes approved crops.
- The [Telethon importer](../src/cancel_capture/adapters/telethon_history.py) backfills existing
  channel images without reposting them.
- The private [Streamlit application](../src/cancel_capture/streamlit_app.py) supports upload,
  reanalysis, catalog inspection, local rejection, and bilingual search. It deliberately does not
  publish to the channel.

## Ingestion pipeline

All three ingestion sources—bot uploads, history imports, and Streamlit uploads—construct an
`IngestRequest` with a stable source key and call the same
[`IngestionService`](../src/cancel_capture/application/ingest.py).

1. The service checks the source key before doing provider work. A prior result is returned with an
   `already_existed` marker, and a database uniqueness race is resolved by loading the winner.
2. [`ContentAddressedAssetStore`](../src/cancel_capture/adapters/filesystem.py) hashes and writes
   the exact input bytes before any decode or external call. Originals and derived crops occupy
   separate SHA-256 paths under the private data directory; writes use a temporary file plus atomic
   replace.
3. [`BestEffortMetadataExtractor`](../src/cancel_capture/adapters/metadata.py) reads namespaced raw
   metadata and normalized archive fields with ExifTool, falling back to Pillow while recording that
   fallback. Host paths and file permissions are removed from ExifTool output.
4. [`PillowImageProcessor`](../src/cancel_capture/adapters/image.py) applies EXIF orientation and
   produces a bounded, metadata-free JPEG for analysis. The archival original remains unchanged.
5. The vision adapter returns an objective scene observation and normalized boxes for every
   qualifying sign. A separate text adapter produces aligned English/Russian photo and sign
   descriptions plus bilingual topics. Structured responses are validated before use.
6. The embedding adapter receives one deterministic search document for the photo and one for each
   sign. Returned count and index order are checked.
7. Each proposed box is expanded and locally refined around red pixels, then applied to the oriented
   full-resolution source. Crops are independently encoded metadata-free JPEG assets.
8. One transaction inserts the photo, its signs, descriptions, detection evidence, embeddings,
   metadata, source provenance, and initial review states. Provider, model, namespace, and embedding
   dimensions stay attached to generated records.

History import uses the same pipeline with two intentional differences: if detection finds no sign,
the imported channel image is treated as a full-frame sign, and created signs start as `published`
with a link to the existing channel message. Per-message failures are durable, later messages still
run, and rerunning the oldest-first import skips completed source keys.

## Review and publication

Every detected sign has its own state and opaque review token. New uploads normally create
`pending_review` signs; the parent photo is immediately `ready`. The
[`ReviewService`](../src/cancel_capture/application/review.py) coordinates conditional catalog
updates with the channel publisher:

```text
pending_review --Publish--> publishing --send and record--> published
       |                          |
       +--Reject--> rejected      +--error or uncertain result--> failed
                                                               |        |
                                                        Retry--+        +--Reject
```

Approval first claims exactly the expected state and token in SQLite. Only the claimant calls
Telegram, so replayed or stale callbacks cannot start another ordinary send. A send error, a local
commit error after Telegram returns success, or a process restart during `publishing` moves the sign
to `failed` with a fresh token and an audit event. Because Telegram has no idempotency key, retry is
always manual after channel inspection. `/markpublished` forwards and verifies a specified existing
channel image before reconciling a failed item as published.

The bot authorizes only the configured numeric owner in a private chat, rejects compressed photo
uploads, enforces the configured download limit before and after download, and checks its channel
posting permission at startup. Review previews and public posts use an in-memory
[`render_telegram_photo`](../src/cancel_capture/adapters/telegram_photo.py) rendition that satisfies
Bot API size, dimension, and aspect-ratio constraints without replacing the archival crop. Public
posts currently contain the crop only; private review cards contain descriptions and a metadata
summary.

## Persistence and retrieval

[`SQLiteCatalog`](../src/cancel_capture/adapters/sqlite_catalog.py) owns schema migration and all
catalog state. The committed schema is at migration 6. Every connection enables foreign keys, WAL,
and a busy timeout; migrations and multi-record ingestion are forward-only transactions. The
catalog stores:

- content-addressed asset records and photo/sign relationships;
- normalized metadata alongside namespaced raw metadata JSON;
- raw and refined detection boxes, confidence, visible text, and topics;
- bilingual descriptions and provider identities;
- Telegram inbound, history, preview, and channel-post provenance;
- review tokens, state, errors, and append-only review events;
- history-import failure attempts; and
- FTS5 search documents plus little-endian float32 embedding blobs.

[`SearchService`](../src/cancel_capture/application/search.py) embeds the bilingual query, selects
only stored vectors with the same provider, namespace, model, and dimensions, and calculates exact
cosine similarity in process. An FTS5 match adds a small fixed lexical bonus. Results may be scoped
to photos or signs and include their review status and asset path.

## Configuration, deployment, and verification

[`AppConfig`](../src/cancel_capture/config.py) loads immutable storage, provider, bot, and history
settings from environment variables. Vision, text, and embedding roles are independently
configurable. A custom endpoint receives a derived non-secret identity namespace unless an explicit
namespace is supplied, preventing vectors from distinct deployments from being mixed accidentally.

The package supports direct `uv` execution and the committed Docker/Compose setup. Runtime assets,
the SQLite database, and the Telethon session live in the configured private data volume.
Operational setup, backups, and failure recovery are covered by the [runbook](runbook.md); design
rationale and privacy boundaries are covered by [architecture](architecture.md).

The committed tests use fakes for external boundaries and cover asset handling, image processing,
metadata, structured provider adapters, ingestion idempotency, search, review conflicts and
recovery, Telegram behavior, history replay, configuration, and Streamlit startup. The
repository-wide local verification command is `./check.sh`, which runs formatting/lint, strict
Pyright, and pytest without live provider calls.
