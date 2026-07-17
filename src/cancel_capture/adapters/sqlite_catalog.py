from __future__ import annotations

import json
import math
import os
import re
import secrets
import sqlite3
import struct
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from cancel_capture.errors import CandidateNotFoundError, DuplicateSourceError, ReviewConflictError
from cancel_capture.models import (
    BilingualText,
    Embedding,
    ImageMetadata,
    IngestedSign,
    IngestionResult,
    ItemEmbedding,
    ItemKind,
    PreparedIngestion,
    ProviderIdentity,
    PublishedMessage,
    ReviewCandidate,
    ReviewStatus,
    SearchDocument,
    SignEmbeddingDocument,
    StoredAsset,
    TelegramFile,
    TelegramMessageRef,
    TelegramMessageRole,
)

_MIGRATION_1 = """
CREATE TABLE assets (
    id TEXT PRIMARY KEY,
    sha256 TEXT UNIQUE NOT NULL,
    relative_path TEXT UNIQUE NOT NULL,
    media_type TEXT NOT NULL,
    byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
    width INTEGER CHECK (width IS NULL OR width > 0),
    height INTEGER CHECK (height IS NULL OR height > 0),
    original_filename TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE items (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('photo', 'sign')),
    asset_id TEXT NOT NULL REFERENCES assets(id),
    parent_photo_id TEXT REFERENCES items(id),
    source_kind TEXT NOT NULL,
    source_key TEXT UNIQUE,
    status TEXT NOT NULL CHECK (
        status IN ('ready', 'pending_review', 'publishing', 'published', 'rejected', 'failed')
    ),
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK (
        (kind = 'photo' AND parent_photo_id IS NULL) OR
        (kind = 'sign' AND parent_photo_id IS NOT NULL)
    )
);

CREATE INDEX items_parent_idx ON items(parent_photo_id);
CREATE INDEX items_status_idx ON items(status);

CREATE TABLE metadata (
    item_id TEXT PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    raw_json TEXT NOT NULL CHECK (json_valid(raw_json)),
    captured_at TEXT,
    latitude REAL,
    longitude REAL,
    altitude_m REAL,
    camera_make TEXT,
    camera_model TEXT,
    lens_model TEXT,
    software TEXT,
    orientation INTEGER,
    extractor TEXT NOT NULL
);

CREATE TABLE detections (
    sign_item_id TEXT PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    photo_item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    raw_left REAL NOT NULL,
    raw_top REAL NOT NULL,
    raw_right REAL NOT NULL,
    raw_bottom REAL NOT NULL,
    crop_left REAL NOT NULL,
    crop_top REAL NOT NULL,
    crop_right REAL NOT NULL,
    crop_bottom REAL NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    factual_summary TEXT NOT NULL,
    visible_text_json TEXT NOT NULL CHECK (json_valid(visible_text_json)),
    topics_en_json TEXT NOT NULL CHECK (json_valid(topics_en_json)),
    topics_ru_json TEXT NOT NULL CHECK (json_valid(topics_ru_json)),
    UNIQUE(photo_item_id, ordinal)
);

CREATE TABLE descriptions (
    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    language TEXT NOT NULL CHECK (language IN ('en', 'ru')),
    text TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(item_id, language)
);

CREATE TABLE search_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT UNIQUE NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE search_fts USING fts5(
    text,
    content='search_documents',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER search_documents_ai AFTER INSERT ON search_documents BEGIN
    INSERT INTO search_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER search_documents_ad AFTER DELETE ON search_documents BEGIN
    INSERT INTO search_fts(search_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER search_documents_au AFTER UPDATE ON search_documents BEGIN
    INSERT INTO search_fts(search_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO search_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE embeddings (
    search_document_id INTEGER PRIMARY KEY REFERENCES search_documents(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    vector BLOB NOT NULL,
    norm REAL NOT NULL CHECK (norm >= 0),
    created_at TEXT NOT NULL
);

CREATE INDEX embeddings_identity_idx ON embeddings(provider, model);

CREATE TABLE telegram_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT REFERENCES items(id) ON DELETE SET NULL,
    role TEXT NOT NULL CHECK (role IN ('inbound', 'history', 'preview', 'channel_post')),
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    media_group_id TEXT,
    file_id TEXT,
    file_unique_id TEXT,
    file_name TEXT,
    mime_type TEXT,
    file_size INTEGER,
    caption TEXT,
    sent_at TEXT,
    edited_at TEXT,
    UNIQUE(chat_id, message_id, role)
);

CREATE INDEX telegram_messages_item_idx ON telegram_messages(item_id);

CREATE TABLE review_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sign_item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    actor_user_id INTEGER,
    details_json TEXT NOT NULL CHECK (json_valid(details_json)),
    created_at TEXT NOT NULL
);
"""

_MIGRATION_2 = """
DROP INDEX embeddings_identity_idx;
CREATE INDEX embeddings_identity_idx ON embeddings(provider, model, dimensions);
"""

_MIGRATION_3 = """
ALTER TABLE telegram_messages RENAME TO telegram_messages_old;
DROP INDEX telegram_messages_item_idx;

CREATE TABLE telegram_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT REFERENCES items(id) ON DELETE SET NULL,
    role TEXT NOT NULL CHECK (role IN ('inbound', 'history', 'preview', 'channel_post')),
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    media_group_id TEXT,
    file_id TEXT,
    file_unique_id TEXT,
    file_name TEXT,
    mime_type TEXT,
    file_size INTEGER,
    caption TEXT,
    sent_at TEXT,
    edited_at TEXT,
    UNIQUE(item_id, chat_id, message_id, role)
);

INSERT INTO telegram_messages(
    id, item_id, role, chat_id, message_id, media_group_id, file_id, file_unique_id,
    file_name, mime_type, file_size, caption, sent_at, edited_at
)
SELECT id, item_id, role, chat_id, message_id, media_group_id, file_id, file_unique_id,
       file_name, mime_type, file_size, caption, sent_at, edited_at
FROM telegram_messages_old;

DROP TABLE telegram_messages_old;
CREATE INDEX telegram_messages_item_idx ON telegram_messages(item_id);
"""

_MIGRATION_4 = """
CREATE TABLE history_import_failures (
    source_key TEXT PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    error_type TEXT NOT NULL,
    error_text TEXT NOT NULL,
    attempts INTEGER NOT NULL CHECK (attempts > 0),
    first_failed_at TEXT NOT NULL,
    last_failed_at TEXT NOT NULL
);
"""

_MIGRATION_5 = """
ALTER TABLE items ADD COLUMN review_token TEXT;
UPDATE items SET review_token = lower(hex(randomblob(8))) WHERE kind = 'sign';
CREATE UNIQUE INDEX items_review_token_idx ON items(review_token) WHERE review_token IS NOT NULL;
"""

_MIGRATION_6 = """
ALTER TABLE detections ADD COLUMN provider_namespace TEXT NOT NULL DEFAULT 'default';
ALTER TABLE descriptions ADD COLUMN provider_namespace TEXT NOT NULL DEFAULT 'default';
ALTER TABLE embeddings ADD COLUMN provider_namespace TEXT NOT NULL DEFAULT 'default';
DROP INDEX embeddings_identity_idx;
CREATE INDEX embeddings_identity_idx
    ON embeddings(provider, provider_namespace, model, dimensions);
"""

_MIGRATION_7 = """
CREATE TABLE visual_embeddings (
    item_id TEXT PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    provider_namespace TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    vector BLOB NOT NULL,
    norm REAL NOT NULL CHECK (norm >= 0),
    created_at TEXT NOT NULL
);

CREATE INDEX visual_embeddings_identity_idx
    ON visual_embeddings(provider, provider_namespace, model, dimensions);
"""

_MIGRATIONS = (
    (1, _MIGRATION_1),
    (2, _MIGRATION_2),
    (3, _MIGRATION_3),
    (4, _MIGRATION_4),
    (5, _MIGRATION_5),
    (6, _MIGRATION_6),
    (7, _MIGRATION_7),
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _new_review_token() -> str:
    return secrets.token_hex(8)


def _pack_embedding(values: tuple[float, ...]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_embedding(data: bytes, dimensions: int) -> tuple[float, ...]:
    expected = dimensions * 4
    if len(data) != expected:
        raise ValueError(f"Stored embedding has {len(data)} bytes; expected {expected}")
    return tuple(struct.unpack(f"<{dimensions}f", data))


def _json_tuple(value: str) -> tuple[str, ...]:
    decoded = cast(object, json.loads(value))
    if not isinstance(decoded, list):
        raise ValueError("Stored JSON value is not a string list")
    values = cast(list[object], decoded)
    if not all(isinstance(item, str) for item in values):
        raise ValueError("Stored JSON value is not a string list")
    return tuple(cast(list[str], values))


def _optional_str(row: sqlite3.Row, key: str) -> str | None:
    value = cast(object, row[key])
    return value if isinstance(value, str) else None


def _optional_float(row: sqlite3.Row, key: str) -> float | None:
    value = cast(object, row[key])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_int(row: sqlite3.Row, key: str) -> int | None:
    value = cast(object, row[key])
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _execute_migration(connection: sqlite3.Connection, script: str) -> None:
    lines: list[str] = []
    for line in script.splitlines():
        lines.append(line)
        statement = "\n".join(lines).strip()
        if statement and sqlite3.complete_statement(statement):
            connection.execute(statement)
            lines.clear()
    if "\n".join(lines).strip():
        raise RuntimeError("Migration contains an incomplete SQL statement")


class SQLiteCatalog:
    def __init__(self, path: Path) -> None:
        self._path = path

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._path.parent.chmod(0o700)
        connection = sqlite3.connect(self._path, timeout=30)
        os.chmod(self._path, 0o600)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL
                    )
                    """
                )
                rows = connection.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                ).fetchall()
                applied = {cast(int, row["version"]) for row in rows}
                known = {version for version, _script in _MIGRATIONS}
                unknown = applied - known
                if unknown:
                    versions = ", ".join(str(version) for version in sorted(unknown))
                    raise RuntimeError(
                        f"Database contains unsupported schema migration(s): {versions}"
                    )
                for version, script in _MIGRATIONS:
                    if version in applied:
                        continue
                    if any(applied_version > version for applied_version in applied):
                        raise RuntimeError(
                            f"Database is missing schema migration {version} before a later version"
                        )
                    _execute_migration(connection, script)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                        (version, _now()),
                    )
                    applied.add(version)
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def find_ingestion(self, source_key: str) -> IngestionResult | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT i.id, a.relative_path, de.text AS description_en,
                       dr.text AS description_ru,
                       m.raw_json, m.captured_at, m.latitude, m.longitude, m.altitude_m,
                       m.camera_make, m.camera_model, m.lens_model, m.software,
                       m.orientation, m.extractor
                FROM items i
                JOIN assets a ON a.id = i.asset_id
                JOIN descriptions de ON de.item_id = i.id AND de.language = 'en'
                JOIN descriptions dr ON dr.item_id = i.id AND dr.language = 'ru'
                JOIN metadata m ON m.item_id = i.id
                WHERE i.kind = 'photo' AND i.source_key = ?
                """,
                (source_key,),
            ).fetchone()
            if row is None:
                return None
            photo_item_id = cast(str, row["id"])
            signs = self._load_ingested_signs(connection, photo_item_id)
            return IngestionResult(
                photo_item_id=photo_item_id,
                original_relative_path=cast(str, row["relative_path"]),
                description=BilingualText(
                    en=cast(str, row["description_en"]),
                    ru=cast(str, row["description_ru"]),
                ),
                metadata=self._metadata_from_row(row),
                signs=signs,
            )

    def record_history_import_failure(
        self,
        source_key: str,
        chat_id: int,
        message_id: int,
        error: Exception,
    ) -> None:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO history_import_failures(
                    source_key, chat_id, message_id, error_type, error_text, attempts,
                    first_failed_at, last_failed_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(source_key) DO UPDATE SET
                    error_type = excluded.error_type,
                    error_text = excluded.error_text,
                    attempts = history_import_failures.attempts + 1,
                    last_failed_at = excluded.last_failed_at
                """,
                (
                    source_key,
                    chat_id,
                    message_id,
                    type(error).__name__,
                    str(error)[:2000],
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()

    def clear_history_import_failure(self, source_key: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM history_import_failures WHERE source_key = ?", (source_key,)
            )
            connection.commit()

    def insert_ingestion(self, ingestion: PreparedIngestion) -> IngestionResult:
        timestamp = _now()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._insert_asset(connection, ingestion.asset, timestamp)
                connection.execute(
                    """
                    INSERT INTO items(
                        id, kind, asset_id, parent_photo_id, source_kind, source_key, status,
                        created_at, updated_at
                    ) VALUES (?, 'photo', ?, NULL, ?, ?, 'ready', ?, ?)
                    """,
                    (
                        ingestion.photo_item_id,
                        ingestion.asset.sha256,
                        ingestion.source_kind.value,
                        ingestion.source_key,
                        timestamp,
                        timestamp,
                    ),
                )
                self._insert_metadata(connection, ingestion.photo_item_id, ingestion.metadata)
                self._insert_descriptions(
                    connection,
                    ingestion.photo_item_id,
                    ingestion.description,
                    ingestion.text_identity,
                    timestamp,
                )
                self._insert_document(
                    connection,
                    ingestion.photo_item_id,
                    ingestion.description.search_text(),
                    ingestion.embedding,
                    timestamp,
                )
                if ingestion.source_message is not None:
                    self._insert_telegram_message(
                        connection,
                        ingestion.photo_item_id,
                        ingestion.source_message,
                    )

                for sign in ingestion.signs:
                    self._insert_asset(connection, sign.asset, timestamp)
                    connection.execute(
                        """
                        INSERT INTO items(
                            id, kind, asset_id, parent_photo_id, source_kind, source_key, status,
                            review_token, created_at, updated_at
                        ) VALUES (?, 'sign', ?, ?, ?, NULL, ?, ?, ?, ?)
                        """,
                        (
                            sign.item_id,
                            sign.asset.sha256,
                            ingestion.photo_item_id,
                            ingestion.source_kind.value,
                            sign.status.value,
                            _new_review_token(),
                            timestamp,
                            timestamp,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO detections(
                            sign_item_id, photo_item_id, ordinal,
                            raw_left, raw_top, raw_right, raw_bottom,
                            crop_left, crop_top, crop_right, crop_bottom,
                            confidence, provider, model, factual_summary, visible_text_json,
                            topics_en_json, topics_ru_json, provider_namespace
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            sign.item_id,
                            ingestion.photo_item_id,
                            sign.observation.ordinal,
                            sign.observation.box.left,
                            sign.observation.box.top,
                            sign.observation.box.right,
                            sign.observation.box.bottom,
                            sign.crop_box.left,
                            sign.crop_box.top,
                            sign.crop_box.right,
                            sign.crop_box.bottom,
                            sign.observation.confidence,
                            ingestion.vision_identity.provider,
                            ingestion.vision_identity.model,
                            sign.observation.factual_summary,
                            json.dumps(sign.observation.visible_text, ensure_ascii=False),
                            json.dumps(sign.description.topics_en, ensure_ascii=False),
                            json.dumps(sign.description.topics_ru, ensure_ascii=False),
                            ingestion.vision_identity.namespace,
                        ),
                    )
                    self._insert_descriptions(
                        connection,
                        sign.item_id,
                        sign.description.text,
                        ingestion.text_identity,
                        timestamp,
                    )
                    sign_search_text = (
                        f"{sign.description.text.search_text()}\n\n"
                        f"Topics (English): {', '.join(sign.description.topics_en)}\n"
                        f"Темы (Русский): {', '.join(sign.description.topics_ru)}"
                    )
                    self._insert_document(
                        connection,
                        sign.item_id,
                        sign_search_text,
                        sign.embedding,
                        timestamp,
                    )
                    if sign.published_message is not None:
                        self._insert_telegram_message(
                            connection,
                            sign.item_id,
                            sign.published_message,
                        )
                        self._insert_review_event(
                            connection,
                            sign.item_id,
                            "imported_published",
                            None,
                            {},
                            timestamp,
                        )
                connection.commit()
        except sqlite3.IntegrityError as error:
            if self.find_ingestion(ingestion.source_key) is not None:
                raise DuplicateSourceError(ingestion.source_key) from error
            raise
        result = self.find_ingestion(ingestion.source_key)
        if result is None:
            raise RuntimeError("Inserted ingestion could not be loaded")
        return result

    def get_candidate(self, item_id: str) -> ReviewCandidate:
        with self._connect() as connection:
            row = self._candidate_query(connection, "s.id = ?", (item_id,)).fetchone()
            if row is None:
                raise CandidateNotFoundError(item_id)
            return self._candidate_from_row(row)

    def list_candidates(self, status: ReviewStatus | None = None) -> tuple[ReviewCandidate, ...]:
        with self._connect() as connection:
            if status is None:
                rows = self._candidate_query(connection, "1 = 1", ()).fetchall()
            else:
                rows = self._candidate_query(connection, "s.status = ?", (status.value,)).fetchall()
            return tuple(self._candidate_from_row(row) for row in rows)

    def list_sign_embedding_documents(self) -> tuple[SignEmbeddingDocument, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT i.id, i.parent_photo_id, i.status, a.relative_path,
                       de.text AS description_en, dr.text AS description_ru,
                       d.topics_en_json, d.topics_ru_json,
                       se.provider AS semantic_provider,
                       se.provider_namespace AS semantic_namespace,
                       se.model AS semantic_model,
                       se.dimensions AS semantic_dimensions,
                       se.vector AS semantic_vector,
                       ve.provider AS visual_provider,
                       ve.provider_namespace AS visual_namespace,
                       ve.model AS visual_model,
                       ve.dimensions AS visual_dimensions,
                       ve.vector AS visual_vector
                FROM items i
                JOIN assets a ON a.id = i.asset_id
                JOIN descriptions de ON de.item_id = i.id AND de.language = 'en'
                JOIN descriptions dr ON dr.item_id = i.id AND dr.language = 'ru'
                JOIN detections d ON d.sign_item_id = i.id
                JOIN search_documents sd ON sd.item_id = i.id
                JOIN embeddings se ON se.search_document_id = sd.id
                LEFT JOIN visual_embeddings ve ON ve.item_id = i.id
                WHERE i.kind = 'sign'
                ORDER BY i.id
                """
            ).fetchall()
        documents: list[SignEmbeddingDocument] = []
        for row in rows:
            semantic_dimensions = cast(int, row["semantic_dimensions"])
            semantic_identity = ProviderIdentity(
                provider=cast(str, row["semantic_provider"]),
                namespace=cast(str, row["semantic_namespace"]),
                model=cast(str, row["semantic_model"]),
            )
            visual_embedding: Embedding | None = None
            visual_dimensions = _optional_int(row, "visual_dimensions")
            visual_vector = cast(object, row["visual_vector"])
            if visual_dimensions is not None and isinstance(visual_vector, bytes):
                visual_embedding = Embedding(
                    identity=ProviderIdentity(
                        provider=cast(str, row["visual_provider"]),
                        namespace=cast(str, row["visual_namespace"]),
                        model=cast(str, row["visual_model"]),
                    ),
                    values=_unpack_embedding(visual_vector, visual_dimensions),
                )
            documents.append(
                SignEmbeddingDocument(
                    item_id=cast(str, row["id"]),
                    parent_photo_id=cast(str, row["parent_photo_id"]),
                    text=BilingualText(
                        en=cast(str, row["description_en"]),
                        ru=cast(str, row["description_ru"]),
                    ),
                    topics_en=_json_tuple(cast(str, row["topics_en_json"])),
                    topics_ru=_json_tuple(cast(str, row["topics_ru_json"])),
                    asset_relative_path=cast(str, row["relative_path"]),
                    status=ReviewStatus(cast(str, row["status"])),
                    semantic_embedding=Embedding(
                        identity=semantic_identity,
                        values=_unpack_embedding(
                            cast(bytes, row["semantic_vector"]), semantic_dimensions
                        ),
                    ),
                    visual_embedding=visual_embedding,
                )
            )
        return tuple(documents)

    def upsert_visual_embeddings(self, embeddings: tuple[ItemEmbedding, ...]) -> None:
        if not embeddings:
            return
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for embedding in embeddings:
                self._insert_visual_embedding(connection, embedding, timestamp)
            connection.commit()

    def record_preview(self, item_id: str, message: TelegramMessageRef) -> None:
        if message.role is not TelegramMessageRole.PREVIEW:
            raise ValueError("Preview records require the preview Telegram role")
        with self._connect() as connection:
            self._insert_telegram_message(connection, item_id, message)
            connection.commit()

    def has_preview(self, item_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM telegram_messages
                WHERE item_id = ? AND role = 'preview'
                LIMIT 1
                """,
                (item_id,),
            ).fetchone()
        return row is not None

    def claim_publish(
        self,
        item_id: str,
        review_token: str,
        expected_status: ReviewStatus,
        actor_user_id: int,
    ) -> ReviewCandidate:
        if expected_status not in (ReviewStatus.PENDING, ReviewStatus.FAILED):
            raise ValueError("Only pending or failed candidates can be claimed")
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE items
                SET status = 'publishing', last_error = NULL, updated_at = ?
                WHERE id = ? AND kind = 'sign' AND status = ? AND review_token = ?
                """,
                (timestamp, item_id, expected_status.value, review_token),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ReviewConflictError(f"Candidate {item_id} is not publishable")
            self._insert_review_event(
                connection, item_id, "publish_claimed", actor_user_id, {}, timestamp
            )
            connection.commit()
        return self.get_candidate(item_id)

    def recover_interrupted_publishes(self) -> int:
        timestamp = _now()
        reason = (
            "The process stopped while publishing, so the Telegram outcome is uncertain. "
            "Check the channel before retrying."
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT id FROM items WHERE kind = 'sign' AND status = 'publishing'"
            ).fetchall()
            item_ids = tuple(cast(str, row["id"]) for row in rows)
            if item_ids:
                connection.executemany(
                    """
                    UPDATE items
                    SET status = 'failed', last_error = ?, review_token = ?, updated_at = ?
                    WHERE id = ? AND kind = 'sign' AND status = 'publishing'
                    """,
                    ((reason, _new_review_token(), timestamp, item_id) for item_id in item_ids),
                )
                for item_id in item_ids:
                    self._insert_review_event(
                        connection,
                        item_id,
                        "publish_interrupted",
                        None,
                        {"error": reason},
                        timestamp,
                    )
            connection.commit()
        return len(item_ids)

    def complete_publish(self, item_id: str, actor_user_id: int, message: PublishedMessage) -> None:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE items
                SET status = 'published', last_error = NULL, updated_at = ?
                WHERE id = ? AND kind = 'sign' AND status = 'publishing'
                """,
                (timestamp, item_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ReviewConflictError(f"Candidate {item_id} is not being published")
            self._insert_telegram_message(
                connection,
                item_id,
                TelegramMessageRef(
                    role=TelegramMessageRole.CHANNEL_POST,
                    chat_id=message.chat_id,
                    message_id=message.message_id,
                    sent_at=message.sent_at,
                    file=message.file,
                ),
            )
            self._insert_review_event(
                connection,
                item_id,
                "published",
                actor_user_id,
                {"chat_id": message.chat_id, "message_id": message.message_id},
                timestamp,
            )
            connection.commit()

    def reconcile_publish(
        self, item_id: str, actor_user_id: int, message: PublishedMessage
    ) -> None:
        timestamp = _now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE items
                SET status = 'published', last_error = NULL, updated_at = ?
                WHERE id = ? AND kind = 'sign' AND status = 'failed'
                """,
                (timestamp, item_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise ReviewConflictError(f"Candidate {item_id} is not reconcilable")
            self._insert_telegram_message(
                connection,
                item_id,
                TelegramMessageRef(
                    role=TelegramMessageRole.CHANNEL_POST,
                    chat_id=message.chat_id,
                    message_id=message.message_id,
                    sent_at=message.sent_at,
                    file=message.file,
                ),
            )
            self._insert_review_event(
                connection,
                item_id,
                "publish_reconciled",
                actor_user_id,
                {"chat_id": message.chat_id, "message_id": message.message_id},
                timestamp,
            )
            connection.commit()

    def fail_publish(self, item_id: str, actor_user_id: int, error: str) -> None:
        timestamp = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE items
                SET status = 'failed', last_error = ?, review_token = ?, updated_at = ?
                WHERE id = ? AND kind = 'sign' AND status = 'publishing'
                """,
                (error[:2000], _new_review_token(), timestamp, item_id),
            )
            if cursor.rowcount != 1:
                raise ReviewConflictError(f"Candidate {item_id} is not being published")
            self._insert_review_event(
                connection,
                item_id,
                "publish_failed",
                actor_user_id,
                {"error": error[:2000]},
                timestamp,
            )
            connection.commit()

    def reject(self, item_id: str, review_token: str, actor_user_id: int) -> ReviewCandidate:
        timestamp = _now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE items
                SET status = 'rejected', last_error = NULL, updated_at = ?
                WHERE id = ? AND kind = 'sign'
                  AND status IN ('pending_review', 'failed') AND review_token = ?
                """,
                (timestamp, item_id, review_token),
            )
            if cursor.rowcount != 1:
                raise ReviewConflictError(f"Candidate {item_id} is not rejectable")
            self._insert_review_event(connection, item_id, "rejected", actor_user_id, {}, timestamp)
            connection.commit()
        return self.get_candidate(item_id)

    def list_search_documents(
        self, identity: ProviderIdentity, dimensions: int, kind: str | None = None
    ) -> tuple[SearchDocument, ...]:
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        parameters: list[object] = [
            identity.provider,
            identity.namespace,
            identity.model,
            dimensions,
        ]
        kind_clause = ""
        if kind is not None:
            kind_clause = " AND i.kind = ?"
            parameters.append(kind)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT i.id, i.kind, i.status, a.relative_path,
                       de.text AS description_en, dr.text AS description_ru,
                       e.dimensions, e.vector
                FROM search_documents sd
                JOIN embeddings e ON e.search_document_id = sd.id
                JOIN items i ON i.id = sd.item_id
                JOIN assets a ON a.id = i.asset_id
                JOIN descriptions de ON de.item_id = i.id AND de.language = 'en'
                JOIN descriptions dr ON dr.item_id = i.id AND dr.language = 'ru'
                WHERE e.provider = ? AND e.provider_namespace = ?
                  AND e.model = ? AND e.dimensions = ? {kind_clause}
                """,
                tuple(parameters),
            ).fetchall()
        documents: list[SearchDocument] = []
        for row in rows:
            dimensions = cast(int, row["dimensions"])
            vector = _unpack_embedding(cast(bytes, row["vector"]), dimensions)
            documents.append(
                SearchDocument(
                    item_id=cast(str, row["id"]),
                    kind=ItemKind(cast(str, row["kind"])),
                    text=BilingualText(
                        en=cast(str, row["description_en"]),
                        ru=cast(str, row["description_ru"]),
                    ),
                    asset_relative_path=cast(str, row["relative_path"]),
                    embedding=Embedding(identity=identity, values=vector),
                    status=ReviewStatus(cast(str, row["status"])),
                )
            )
        return tuple(documents)

    def lexical_matches(self, query: str, kind: str | None = None) -> frozenset[str]:
        tokens = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
        if not tokens:
            return frozenset()
        expression = " OR ".join(f'"{token}"' for token in tokens[:20])
        parameters: list[object] = [expression]
        kind_clause = ""
        if kind is not None:
            kind_clause = " AND i.kind = ?"
            parameters.append(kind)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT i.id
                FROM search_fts
                JOIN search_documents sd ON sd.id = search_fts.rowid
                JOIN items i ON i.id = sd.item_id
                WHERE search_fts MATCH ? {kind_clause}
                LIMIT 200
                """,
                tuple(parameters),
            ).fetchall()
        return frozenset(cast(str, row["id"]) for row in rows)

    @staticmethod
    def _insert_asset(connection: sqlite3.Connection, asset: StoredAsset, timestamp: str) -> None:
        connection.execute(
            """
            INSERT INTO assets(
                id, sha256, relative_path, media_type, byte_size, width, height,
                original_filename, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                width = COALESCE(assets.width, excluded.width),
                height = COALESCE(assets.height, excluded.height),
                original_filename = COALESCE(assets.original_filename, excluded.original_filename)
            """,
            (
                asset.sha256,
                asset.sha256,
                asset.relative_path,
                asset.media_type,
                asset.byte_size,
                asset.width,
                asset.height,
                asset.original_filename,
                timestamp,
            ),
        )

    @staticmethod
    def _insert_metadata(
        connection: sqlite3.Connection, item_id: str, metadata: ImageMetadata
    ) -> None:
        connection.execute(
            """
            INSERT INTO metadata(
                item_id, raw_json, captured_at, latitude, longitude, altitude_m,
                camera_make, camera_model, lens_model, software, orientation, extractor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                metadata.raw_json,
                metadata.captured_at,
                metadata.latitude,
                metadata.longitude,
                metadata.altitude_m,
                metadata.camera_make,
                metadata.camera_model,
                metadata.lens_model,
                metadata.software,
                metadata.orientation,
                metadata.extractor,
            ),
        )

    @staticmethod
    def _insert_descriptions(
        connection: sqlite3.Connection,
        item_id: str,
        text: BilingualText,
        identity: ProviderIdentity,
        timestamp: str,
    ) -> None:
        connection.executemany(
            """
            INSERT INTO descriptions(
                item_id, language, text, provider, model, created_at, provider_namespace
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    item_id,
                    "en",
                    text.en,
                    identity.provider,
                    identity.model,
                    timestamp,
                    identity.namespace,
                ),
                (
                    item_id,
                    "ru",
                    text.ru,
                    identity.provider,
                    identity.model,
                    timestamp,
                    identity.namespace,
                ),
            ),
        )

    @staticmethod
    def _insert_document(
        connection: sqlite3.Connection,
        item_id: str,
        text: str,
        embedding: Embedding,
        timestamp: str,
    ) -> None:
        cursor = connection.execute(
            "INSERT INTO search_documents(item_id, text, created_at) VALUES (?, ?, ?)",
            (item_id, text, timestamp),
        )
        document_id = cursor.lastrowid
        if document_id is None:
            raise RuntimeError("SQLite did not return a search document ID")
        norm = math.sqrt(sum(value * value for value in embedding.values))
        connection.execute(
            """
            INSERT INTO embeddings(
                search_document_id, provider, model, dimensions, vector, norm, created_at,
                provider_namespace
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                embedding.identity.provider,
                embedding.identity.model,
                len(embedding.values),
                _pack_embedding(embedding.values),
                norm,
                timestamp,
                embedding.identity.namespace,
            ),
        )

    @staticmethod
    def _insert_visual_embedding(
        connection: sqlite3.Connection,
        item_embedding: ItemEmbedding,
        timestamp: str,
    ) -> None:
        embedding = item_embedding.embedding
        norm = math.sqrt(sum(value * value for value in embedding.values))
        connection.execute(
            """
            INSERT INTO visual_embeddings(
                item_id, provider, provider_namespace, model, dimensions, vector, norm, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                provider = excluded.provider,
                provider_namespace = excluded.provider_namespace,
                model = excluded.model,
                dimensions = excluded.dimensions,
                vector = excluded.vector,
                norm = excluded.norm,
                created_at = excluded.created_at
            """,
            (
                item_embedding.item_id,
                embedding.identity.provider,
                embedding.identity.namespace,
                embedding.identity.model,
                len(embedding.values),
                _pack_embedding(embedding.values),
                norm,
                timestamp,
            ),
        )

    @staticmethod
    def _insert_telegram_message(
        connection: sqlite3.Connection, item_id: str, message: TelegramMessageRef
    ) -> None:
        file = message.file or TelegramFile()
        connection.execute(
            """
            INSERT OR IGNORE INTO telegram_messages(
                item_id, role, chat_id, message_id, media_group_id, file_id, file_unique_id,
                file_name, mime_type, file_size, caption, sent_at, edited_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                message.role.value,
                message.chat_id,
                message.message_id,
                message.media_group_id,
                file.file_id,
                file.file_unique_id,
                file.file_name,
                file.mime_type,
                file.file_size,
                message.caption,
                message.sent_at,
                message.edited_at,
            ),
        )

    @staticmethod
    def _insert_review_event(
        connection: sqlite3.Connection,
        item_id: str,
        action: str,
        actor_user_id: int | None,
        details: dict[str, object],
        timestamp: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO review_events(
                sign_item_id, action, actor_user_id, details_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                item_id,
                action,
                actor_user_id,
                json.dumps(details, ensure_ascii=False, separators=(",", ":")),
                timestamp,
            ),
        )

    @staticmethod
    def _load_ingested_signs(
        connection: sqlite3.Connection, photo_item_id: str
    ) -> tuple[IngestedSign, ...]:
        rows = connection.execute(
            """
            SELECT s.id, s.status, d.ordinal, a.relative_path,
                   de.text AS description_en, dr.text AS description_ru
            FROM items s
            JOIN detections d ON d.sign_item_id = s.id
            JOIN assets a ON a.id = s.asset_id
            JOIN descriptions de ON de.item_id = s.id AND de.language = 'en'
            JOIN descriptions dr ON dr.item_id = s.id AND dr.language = 'ru'
            WHERE s.parent_photo_id = ?
            ORDER BY d.ordinal
            """,
            (photo_item_id,),
        ).fetchall()
        return tuple(
            IngestedSign(
                item_id=cast(str, row["id"]),
                ordinal=cast(int, row["ordinal"]),
                status=ReviewStatus(cast(str, row["status"])),
                crop_relative_path=cast(str, row["relative_path"]),
                description=BilingualText(
                    en=cast(str, row["description_en"]),
                    ru=cast(str, row["description_ru"]),
                ),
            )
            for row in rows
        )

    @staticmethod
    def _candidate_query(
        connection: sqlite3.Connection, condition: str, parameters: tuple[object, ...]
    ) -> sqlite3.Cursor:
        return connection.execute(
            f"""
            SELECT s.id, s.review_token, s.status, s.last_error, d.ordinal, d.visible_text_json,
                   d.topics_en_json, d.topics_ru_json, a.relative_path,
                   pde.text AS photo_description_en, pdr.text AS photo_description_ru,
                   sde.text AS sign_description_en, sdr.text AS sign_description_ru,
                   m.raw_json, m.captured_at, m.latitude, m.longitude, m.altitude_m,
                   m.camera_make, m.camera_model, m.lens_model, m.software,
                   m.orientation, m.extractor
            FROM items s
            JOIN detections d ON d.sign_item_id = s.id
            JOIN assets a ON a.id = s.asset_id
            JOIN items p ON p.id = s.parent_photo_id
            JOIN descriptions pde ON pde.item_id = p.id AND pde.language = 'en'
            JOIN descriptions pdr ON pdr.item_id = p.id AND pdr.language = 'ru'
            JOIN descriptions sde ON sde.item_id = s.id AND sde.language = 'en'
            JOIN descriptions sdr ON sdr.item_id = s.id AND sdr.language = 'ru'
            JOIN metadata m ON m.item_id = p.id
            WHERE {condition}
            ORDER BY s.created_at, d.ordinal
            """,
            parameters,
        )

    @staticmethod
    def _candidate_from_row(row: sqlite3.Row) -> ReviewCandidate:
        return ReviewCandidate(
            item_id=cast(str, row["id"]),
            review_token=cast(str, row["review_token"]),
            ordinal=cast(int, row["ordinal"]),
            status=ReviewStatus(cast(str, row["status"])),
            crop_relative_path=cast(str, row["relative_path"]),
            photo_description=BilingualText(
                en=cast(str, row["photo_description_en"]),
                ru=cast(str, row["photo_description_ru"]),
            ),
            sign_description=BilingualText(
                en=cast(str, row["sign_description_en"]),
                ru=cast(str, row["sign_description_ru"]),
            ),
            topics_en=_json_tuple(cast(str, row["topics_en_json"])),
            topics_ru=_json_tuple(cast(str, row["topics_ru_json"])),
            visible_text=_json_tuple(cast(str, row["visible_text_json"])),
            metadata=SQLiteCatalog._metadata_from_row(row),
            last_error=_optional_str(row, "last_error"),
        )

    @staticmethod
    def _metadata_from_row(row: sqlite3.Row) -> ImageMetadata:
        return ImageMetadata(
            raw_json=cast(str, row["raw_json"]),
            captured_at=_optional_str(row, "captured_at"),
            latitude=_optional_float(row, "latitude"),
            longitude=_optional_float(row, "longitude"),
            altitude_m=_optional_float(row, "altitude_m"),
            camera_make=_optional_str(row, "camera_make"),
            camera_model=_optional_str(row, "camera_model"),
            lens_model=_optional_str(row, "lens_model"),
            software=_optional_str(row, "software"),
            orientation=_optional_int(row, "orientation"),
            extractor=cast(str, row["extractor"]),
        )
