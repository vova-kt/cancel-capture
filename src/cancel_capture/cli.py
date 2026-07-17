from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from collections.abc import Sequence

from cancel_capture.adapters.sqlite_catalog import SQLiteCatalog
from cancel_capture.adapters.telegram_bot import run_bot
from cancel_capture.adapters.telethon_history import TelethonHistoryImporter
from cancel_capture.config import AppConfig
from cancel_capture.container import build_services
from cancel_capture.errors import CancelCaptureError, ConfigurationError
from cancel_capture.models import ItemKind


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cancel-capture")
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor = subcommands.add_parser("doctor", help="initialize storage and validate configuration")
    doctor.add_argument("--quiet", action="store_true")
    subcommands.add_parser("bot", help="run the owner-only Telegram bot")
    subcommands.add_parser(
        "import-history", help="import all existing channel images with Telethon"
    )
    search = subcommands.add_parser("search", help="search descriptions and embeddings")
    search.add_argument("query")
    search.add_argument("--kind", choices=[kind.value for kind in ItemKind])
    search.add_argument("--limit", type=int, default=20)
    return parser


def _doctor(config: AppConfig, *, quiet: bool) -> int:
    config.storage.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    SQLiteCatalog(config.storage.sqlite_path).initialize()
    issues: list[str] = []
    for role, provider in (
        ("vision", config.vision),
        ("text", config.text),
        ("embedding", config.embedding),
    ):
        if provider.api_key is None:
            issues.append(f"{role} provider has no API key")
    try:
        config.bot.validate()
    except ConfigurationError as error:
        issues.append(str(error))
    try:
        config.history.validate()
    except ConfigurationError as error:
        issues.append(str(error))
    if shutil.which("exiftool") is None:
        issues.append("ExifTool is not installed; Pillow metadata fallback will be used")

    if not quiet:
        print(f"Data directory: {config.storage.data_dir.resolve()}")
        print(f"SQLite database: {config.storage.sqlite_path.resolve()}")
        print("Database schema: ready")
        for issue in issues:
            print(f"Notice: {issue}")
    return 0


async def _search(config: AppConfig, query: str, kind: str | None, limit: int) -> int:
    services = build_services(config)
    hits = await services.search.search(
        query,
        kind=ItemKind(kind) if kind is not None else None,
        limit=limit,
    )
    for hit in hits:
        path = services.assets.resolve(hit.asset_relative_path)
        print(f"{hit.score:.4f}\t{hit.kind.value}\t{hit.status.value}\t{path}")
        print(f"  EN: {hit.description.en}")
        print(f"  RU: {hit.description.ru}")
    return 0


def run(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = AppConfig.from_env()
    command = str(arguments.command)
    if command == "doctor":
        return _doctor(config, quiet=bool(arguments.quiet))
    if command == "bot":
        config.bot.validate()
        run_bot(build_services(config))
        return 0
    if command == "import-history":
        config.history.validate()
        importer = TelethonHistoryImporter(build_services(config), config.history)
        summary = asyncio.run(importer.run())
        print(
            f"Import complete: {summary.imported} new, {summary.skipped} already present, "
            f"{summary.failed} failed"
        )
        return 1 if summary.failed else 0
    if command == "search":
        return asyncio.run(
            _search(
                config,
                str(arguments.query),
                str(arguments.kind) if arguments.kind is not None else None,
                int(arguments.limit),
            )
        )
    raise AssertionError(f"Unhandled command: {command}")


def main() -> None:
    try:
        raise SystemExit(run())
    except (CancelCaptureError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
