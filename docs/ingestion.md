# Ingestion pipeline

All three ingestion sources — bot uploads, history imports, and Streamlit uploads — construct an
`IngestRequest` with a stable source key and call the same
[`IngestionService`](../src/cancel_capture/application/ingest.py).

1. The service checks the source key before doing provider work. A prior result is returned with an
   `already_existed` marker, and a database uniqueness race is resolved by loading the winner.
2. [`ContentAddressedAssetStore`](../src/cancel_capture/adapters/filesystem.py) hashes and writes
   the exact input bytes before any decode or external call. Originals and derived crops occupy
   separate SHA-256 paths under the private data directory; writes use a temporary file plus
   atomic replace.
3. [`BestEffortMetadataExtractor`](../src/cancel_capture/adapters/metadata.py) reads namespaced raw
   metadata and normalized archive fields with ExifTool, falling back to Pillow while recording
   that fallback. Host paths and file permissions are removed from ExifTool output.
4. [`PillowImageProcessor`](../src/cancel_capture/adapters/image.py) applies EXIF orientation and
   produces a bounded, metadata-free JPEG for analysis. The archival original remains unchanged.
5. The vision adapter returns an objective scene observation and normalized boxes for every
   qualifying sign. A separate text adapter produces aligned English/Russian photo and sign
   descriptions plus bilingual topics. Structured responses are validated before use.
6. The embedding adapter receives one deterministic search document for the photo and one for
   each sign. Returned count and index order are checked.
7. Each proposed box is expanded and locally refined around red pixels, then applied to the
   oriented full-resolution source. Crops are independently encoded metadata-free JPEG assets.
8. One transaction inserts the photo, its signs, descriptions, detection evidence, embeddings,
   metadata, source provenance, and initial review states. Provider, model, namespace, and
   embedding dimensions stay attached to generated records.

## History import

History import uses the same pipeline with two intentional differences: if detection finds no
sign, the imported channel image is treated as a full-frame sign, and created signs start as
`published` with a link to the existing channel message. Per-message failures are durable, later
messages still run, and rerunning the oldest-first import skips completed source keys.

## Related pages

- [Implementation overview](implementation.md) — index and runtime shape.
- [Review and publication](review.md) — what happens to a sign once it lands in `pending_review`.
- [Persistence and retrieval](persistence.md) — how the transactional insert lands on disk.
