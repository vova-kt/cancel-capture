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

## Streamlit-only experiments: clustering and narrative

The Streamlit UI adds two experiment tabs on top of the ingestion pipeline. They live behind an
optional narrative-provider service graph
([`build_narrative_services`](../src/cancel_capture/container.py)) that is composed lazily so the
bot and CLI never need a narrative API key.

### Clustering

- Hierarchical clustering runs in-process with a deterministic average-linkage cosine implementation
  ([`average_linkage_cosine`](../src/cancel_capture/application/clustering.py)). Ties break on
  sorted member sets so the same inputs always produce the same tree. All merges plus dendrogram
  segment coordinates are computed once per session; the cut is a pure function of the slider.
- Semantic embeddings are read via
  [`SQLiteCatalog.list_sign_embedding_documents`](../src/cancel_capture/adapters/sqlite_catalog.py)
  and converted into `SearchDocument`s so the same clustering surface works for future embedding
  variants. Visual embeddings live in a separate `visual_embeddings` table (schema migration 7),
  refreshed by
  [`VisualEmbeddingService.ensure_current`](../src/cancel_capture/application/visual_embeddings.py)
  through the deterministic
  [`PillowVisualEmbeddingProvider`](../src/cancel_capture/adapters/visual_embedding.py); the visual
  vector is used by the narrative similarity mode, not the clustering view.
- The Streamlit tab renders the dendrogram with Altair rule marks, draws a dashed cut line at the
  selected level, and lays out each cluster as inline HTML thumbnails with `title` tooltips so
  descriptions surface on hover without an extra click. A per-cluster button invokes
  [`ClusterThemeService`](../src/cancel_capture/application/cluster_theme.py) which delegates to the
  structured [`OpenAIClusterThemeProvider`](../src/cancel_capture/adapters/openai_provider.py).

### Narrative

- Anchor selection is either explicit (dropdown) or randomized
  ([`NarrativeSelectionService.random_anchor`](../src/cancel_capture/application/narrative_selection.py)),
  with a seeded `random.Random` so a reroll button remains reproducible during a session. Companion
  selection samples uniformly without replacement from every sibling below a configurable maximum
  cosine similarity, using the current `SimilarityMode` (semantic, visual, or hybrid). Same-photo
  siblings are always excluded so the pool never seeds the story with the anchor's neighbours.
- The
  [`NarrativeExperimentService`](../src/cancel_capture/application/narrative_experiment.py)
  orchestrates: current-events research via
  [`OpenAICurrentNewsProvider`](../src/cancel_capture/adapters/openai_provider.py) (Responses API +
  built-in `web_search` tool), prompt assembly through
  [`render_narrative_user_prompt`](../src/cancel_capture/prompts/narrative.py), and structured
  narrative generation via [`OpenAINarrativeProvider`](../src/cancel_capture/adapters/openai_provider.py).
  The anchor is weighted more heavily than companions by default (2.5 vs. 1.0); the weights and
  reading-minute → target-words mapping are pure functions so alternative strategies can share the
  same orchestration.
- Every generated story is persisted as an atomic, private Markdown artifact under
  `data/narratives/<attempt-id>.md` by
  [`MarkdownNarrativeStore`](../src/cancel_capture/adapters/markdown_narratives.py). Files are
  written with `O_EXCL`, `fsync`, and 0o600 permissions; front matter records prompts, sampling
  parameters, and web citations so historical attempts can be inspected or reused.
- Prompts (vision, archival, cluster theme, narrative system, per-strategy) live under
  [`cancel_capture.prompts`](../src/cancel_capture/prompts/__init__.py) so tuning does not require
  editing adapter code. Test coverage exercises the pure prompt rendering, the store round-trip,
  the selection service, and the structured-output paths of every new provider.
