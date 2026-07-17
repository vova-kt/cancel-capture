"""Backfill visual embeddings for signs that lack a current vector.

Visual embeddings power the narrative selector's HYBRID similarity mode. They're only used
by the Streamlit experiments, so the main ingest path doesn't compute them — this script
runs the Pillow-based provider (CPU-bound feature extraction) over the catalog in a process
pool.

Usage:
    uv run python scripts/backfill_visual_embeddings.py [--workers N] [--batch-size N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from cancel_capture.adapters.image import PillowImageProcessor
from cancel_capture.adapters.visual_embedding import PillowVisualEmbeddingProvider
from cancel_capture.config import AppConfig
from cancel_capture.container import build_services
from cancel_capture.models import Embedding, ItemEmbedding


def _embed_one(payload: tuple[str, str, int]) -> tuple[str, Embedding]:
    item_id, path_str, max_image_pixels = payload
    images = PillowImageProcessor(
        max_image_pixels=max_image_pixels,
        max_analysis_side=4096,
    )
    provider = PillowVisualEmbeddingProvider(max_image_pixels=max_image_pixels)
    prepared = images.prepare(Path(path_str))
    return item_id, provider.embed_one(prepared)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 1),
        help="Number of worker processes (default: cpu_count - 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Rows per SQLite upsert batch (default: 64)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N pending signs (useful for smoke-testing)",
    )
    args = parser.parse_args(argv)

    config = AppConfig.from_env()
    services = build_services(config)
    reference = PillowVisualEmbeddingProvider(max_image_pixels=config.storage.max_image_pixels)

    documents = services.catalog.list_sign_embedding_documents()
    pending = [
        document
        for document in documents
        if document.visual_embedding is None
        or document.visual_embedding.identity != reference.identity
        or len(document.visual_embedding.values) != reference.dimensions
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"{len(pending)}/{len(documents)} signs need a visual embedding")
    if not pending:
        return 0

    payloads: list[tuple[str, str, int]] = [
        (
            document.item_id,
            str(services.assets.resolve(document.asset_relative_path)),
            config.storage.max_image_pixels,
        )
        for document in pending
    ]

    buffer: list[ItemEmbedding] = []
    completed = 0
    failed = 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_embed_one, payload): payload[0] for payload in payloads}
        for future in as_completed(futures):
            item_id = futures[future]
            try:
                _, embedding = future.result()
            except Exception as error:
                failed += 1
                print(f"\n  ! {item_id}: {error}", file=sys.stderr)
                continue
            buffer.append(ItemEmbedding(item_id, embedding))
            completed += 1
            if len(buffer) >= args.batch_size:
                services.catalog.upsert_visual_embeddings(tuple(buffer))
                buffer = []
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed > 0 else 0.0
            print(
                f"\r  {completed}/{len(pending)} embedded ({failed} failed, {rate:.1f}/s)",
                end="",
                flush=True,
            )
    if buffer:
        services.catalog.upsert_visual_embeddings(tuple(buffer))
    print()
    print(f"Done: {completed} embedded, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
