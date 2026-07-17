# Operations runbook

## First deployment

Create a bot with BotFather, add it to `@cancel_capture` as an administrator, and grant posting
permission. Put the bot token and the numeric Telegram user ID—not the mutable username—in `.env`.
Create Telegram API credentials at `my.telegram.org` for the history importer and configure the three
independent provider roles. `uv run cancel-capture doctor` creates the private directories, applies
SQLite migrations, and reports missing role-specific configuration without printing secrets.

Custom provider endpoints receive a non-secret hashed identity automatically. Set a role's explicit
identity namespace only when endpoint aliases should share one deployment identity; never put API
keys or credential-bearing URLs in that field.

The hosted Bot API downloads files up to 20 MB. The bot rejects larger documents before download.
For genuinely larger originals, deploy Telegram's local Bot API server and set both configured Bot API
URLs; do not raise the application limit while still using the hosted endpoint.

## Historical backfill

Run the importer interactively once, preferably before leaving the bot unattended. Its session file
can access the Telegram account and belongs in the data volume with restrictive permissions. Never
copy it into the image, repository, logs, or CI artifacts.

The importer visits messages oldest-first, includes image documents and albums, and commits each
image independently. It can be interrupted and rerun. A bad image is recorded as a durable failure
and does not prevent later messages from importing; the command exits nonzero when any failures need
attention. A historical post time is retained as Telegram provenance, not promoted to camera capture
time. Use provider dashboards to monitor the approximately two model calls plus one embedding batch
per imported image.

## Routine operation

Back up the complete data volume, not just SQLite: database rows reference content-addressed original
and crop paths, and the volume also holds the importer session. Use SQLite's online backup mechanism
or stop the bot briefly before copying the database files; copying only the main database while WAL is
active can produce an incomplete backup.

The Compose named volume is initialized with the image's non-root ownership. If a host bind mount is
used instead, make its directory writable by container UID/GID `999:999`; do not run the bot as root
to work around a permission error.

Keep Streamlit bound to loopback. It exposes private originals, GPS, device metadata, and unpublished
candidates and is a development interface, not a public web application.

## Failure recovery

- A rejected upload remains archived and searchable with its review status; rejection never deletes
  evidence.
- A provider failure leaves the content-addressed original on disk. Re-send the source or rerun the
  idempotent importer after correcting credentials or provider configuration.
- A `failed` publish may be retried manually after checking the channel. Never automate retries for a
  timeout or connection loss that may have occurred after Telegram accepted the photo.
- If inspection confirms an uncertain attempt actually posted, run `/markpublished ARCHIVE_ID
  CHANNEL_MESSAGE_ID` in the private bot chat. The bot first forwards that exact channel image as
  verification, then records the existing post without sending another public photo.
- Bot startup converts an interrupted `publishing` state to `failed` with an audit event. Treat it as
  an uncertain outcome and inspect the channel before pressing Retry.
- If channel permissions change, restore the bot's administrator posting right and retry only after
  confirming that no post was created.
- If search configuration changes embedding provider, model, or dimensions, old vectors remain
  intact but do not compare across that complete identity, which also includes the non-secret
  deployment namespace. Re-embedding should be an explicit maintenance operation.

Run `./check.sh` and build the Docker image before each deployment. CI performs the same deterministic
checks without live Telegram or model credentials.
