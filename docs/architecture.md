# Architecture

## Why the archive separates photos from signs

An uploaded photo is historical context; a detected sign is a searchable claim about one detail in
that context. Keeping both lets later work ask either “what did this street scene contain?” or “show
every sign about bicycles” without discarding composition, location evidence, or neighboring signs.
Every photo and sign therefore has its own bilingual description and embedding, while signs retain a
parent link and both the model-proposed and locally refined crop boxes.

The original asset is written content-addressably before decoding. Analysis uses an EXIF-oriented,
metadata-stripped JPEG so bounding boxes match the pixels being inspected and private GPS/device
fields are not needlessly sent to a model. Refined normalized boxes are applied to the oriented
full-resolution original, not the resized analysis copy. Public crops are independently encoded
JPEGs with no EXIF, XMP, or IPTC data. That full-resolution crop remains the archival asset. Both the
private review preview and channel post use the same deterministic, in-memory delivery rendition:
extreme aspect ratios are padded, dimensions are reduced to the Bot API envelope, and a fixed JPEG
quality ladder keeps the upload below its size limit. The rendition is never persisted or written
back over the crop. Stored paths are relative to the data volume so a Mac archive can move unchanged
to the Linux NUC.

## Replaceable intelligence

Vision, text, and embeddings are separate typed ports. Vision returns objective scene observations,
all qualifying boxes, confidence, and visible text. Text turns those observations into aligned
English/Russian archival descriptions and topic vocabulary. Embedding receives deterministic
bilingual search documents. OpenAI implements all three initially, but each has its own provider,
key, endpoint, and model configuration; a new adapter does not change ingestion or review logic.
Provider identity also includes a non-secret deployment namespace. Custom endpoints derive one from
the endpoint unless explicitly named, preventing vectors from different deployments with the same
model label and dimensions from being compared.

This two-stage design costs an extra small model call, which is acceptable for monthly ingestion and
historical backfill. In return, sign localization can later move to a local detector while description
quality and existing storage contracts remain unchanged.

## Review and delivery state

Each detected sign is independently `pending_review`, `publishing`, `published`, `rejected`, or
`failed`. Approval claims a candidate with one conditional SQLite update before contacting Telegram.
Duplicate button presses therefore cannot start two ordinary sends. Telegram offers no idempotency
key, so a network failure after a send may still be ambiguous; the bot records failure and requires a
human retry instead of silently risking a duplicate channel post. On startup, any publish left in
progress by an interrupted process becomes a failed, explicitly uncertain attempt; the same happens
when Telegram accepts a post but the returned message cannot be committed locally. An owner-only
reconciliation command verifies an existing channel image by forwarding it into the private chat and
then links it, avoiding a duplicate retry.

Historical messages enter as already published and point at their existing channel message. Import
identity comes from the stable numeric channel/message pair rather than filenames, captions, or
Telegram `file_unique_id`. Re-running the importer skips complete sources and is the normal recovery
mechanism after interruption.

## Catalog and retrieval

SQLite owns relational state, audit events, normalized metadata, raw namespaced metadata JSON,
Telegram provenance, FTS5 text, and little-endian float32 embeddings. Foreign keys, WAL, and a busy
timeout are enabled per connection. When ExifTool is configured but fails, the raw metadata records
the failure and the extractor identity makes the Pillow fallback explicit. Search combines exact
cosine similarity with a small lexical FTS boost and returns status and asset provenance alongside
the score.

Exact scanning is simpler and more portable than ANN for hundreds or low thousands of documents.
If narrative experiments or collection growth make it slow, the embedding port and search-document
contract allow an indexed backend without changing the archive.

## Narrative boundary

Future narratives should be generated from explicit catalog selections and keep their prompt,
sources, model identity, and output version separate from factual descriptions. Archival descriptions
remain observational; they must not be retroactively rewritten to support a thesis about increasing
prohibition. This separation preserves the difference between evidence and interpretation.
