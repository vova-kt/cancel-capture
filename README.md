# Cancel Capture

Cancel Capture archives original photographs of prohibition signs, extracts all available image
metadata, detects and crops every round red crossed-out sign, writes detailed English and Russian
descriptions, and stores searchable embeddings in SQLite. A private Telegram review card is created
for each crop; nothing is posted to the channel until the configured owner confirms it.

The current channel is [@cancel_capture](https://t.me/cancel_capture). Narrative generation is
deliberately outside the first release, but the catalog keeps originals, scene context, individual
signs, bilingual descriptions, metadata provenance, and independent embeddings so that later
chronologies or speculative narratives do not require rebuilding the archive.

## Setup

Install Python 3.12–3.14, [uv](https://docs.astral.sh/uv/), and ExifTool
(`brew install exiftool` on macOS), then:

```sh
uv sync --extra dev
cp .env.example .env
./check.sh
```

Fill `.env` with the bot token, numeric owner ID, OpenAI key, and Telegram MTProto credentials.
The bot must be an administrator of the target channel with permission to post. Run a configuration
and database check before starting it:

```sh
uv run cancel-capture doctor
uv run cancel-capture bot
```

Send an image to the bot as a **file/document**, not as a compressed Telegram photo. The bot saves
the exact bytes, analyzes a metadata-stripped oriented copy, and sends one crop and approval card per
detected sign. Each card has independent Publish and Reject buttons.

## Import existing channel photos

The Bot API cannot enumerate old channel history. The importer therefore uses a one-time Telethon
user login and stores its credential-bearing session only in the private data volume:

```sh
uv run cancel-capture import-history
```

The first run asks for the Telegram login code and, when enabled, the account's 2FA password. The
import is idempotent by numeric channel ID and message ID. It downloads every image available in the
channel, treats already-cropped historical images as signs when detection is uncertain, analyzes and
embeds them, and links them to their existing channel posts without reposting.

Telegram commonly strips camera metadata from photos posted to channels. The importer records the
remaining Telegram/file facts and does not mislabel post time as camera capture time.

## Development UI and search

```sh
./streamlit.sh
uv run cancel-capture search "bicycles and scooters"
```

Streamlit supports local upload inspection, explicit same-file reanalysis, local rejection, full
metadata review, catalog browsing, and bilingual semantic search. Public channel publishing remains
behind Telegram confirmation. Exact cosine search over SQLite float32 vectors is intentional at this
archive's scale; it avoids a platform-specific vector extension and can be replaced behind the
catalog interface if the collection grows substantially.

## Docker

```sh
docker compose up -d bot
docker compose --profile dev up streamlit
docker compose --profile tools run --rm importer
```

All originals, crops, SQLite files, and the Telethon session live in the `cancel_capture_data` volume.
Back up that volume; never add it or `.env` to Git.

Documentation:

- [Implementation overview](docs/implementation.md) — entry point; indexes all deep-dive pages and maps components to source
- [Ingestion](docs/ingestion.md) — upload pipeline: detection, cropping, metadata extraction, embedding, and sign analysis
- [Review](docs/review.md) — Telegram approval flow, callback handling, and publish/reject state machine
- [Persistence](docs/persistence.md) — SQLite schema, migrations, WAL setup, and catalog interface
- [Experiments](docs/experiments.md) — Streamlit-only narrative generation and cluster-theme exploration
- [Architecture decisions](docs/architecture.md) — provider protocol design, privacy boundaries, and key tradeoffs
- [Operations runbook](docs/runbook.md) — provider swapping, backup, recovery, and deployment details
