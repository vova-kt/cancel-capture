# Review and publication

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
commit error after Telegram returns success, or a process restart during `publishing` moves the
sign to `failed` with a fresh token and an audit event. Because Telegram has no idempotency key,
retry is always manual after channel inspection. `/markpublished` forwards and verifies a
specified existing channel image before reconciling a failed item as published.

## Bot boundary

The bot authorizes only the configured numeric owner in a private chat, rejects compressed photo
uploads, enforces the configured download limit before and after download, and checks its channel
posting permission at startup. Review previews and public posts use an in-memory
[`render_telegram_photo`](../src/cancel_capture/adapters/telegram_photo.py) rendition that
satisfies Bot API size, dimension, and aspect-ratio constraints without replacing the archival
crop. Public posts currently contain the crop only; private review cards contain descriptions and
a metadata summary.

## Related pages

- [Implementation overview](implementation.md) — index and runtime shape.
- [Ingestion pipeline](ingestion.md) — how a sign reaches `pending_review`.
- [Persistence and retrieval](persistence.md) — how review tokens, publish claims, and Telegram
  provenance are stored.
