from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast

from telethon import TelegramClient  # pyright: ignore[reportMissingTypeStubs]

from cancel_capture.config import TelegramHistoryConfig
from cancel_capture.container import Services
from cancel_capture.models import (
    IngestRequest,
    ReviewStatus,
    SourceKind,
    TelegramFile,
    TelegramMessageRef,
    TelegramMessageRole,
)


class _HistoryFile(Protocol):
    @property
    def name(self) -> str | None: ...

    @property
    def mime_type(self) -> str | None: ...

    @property
    def size(self) -> int | None: ...


class _HistoryMessage(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def chat_id(self) -> int | None: ...

    @property
    def photo(self) -> object | None: ...

    @property
    def document(self) -> object | None: ...

    @property
    def file(self) -> _HistoryFile | None: ...

    @property
    def date(self) -> datetime | None: ...

    @property
    def edit_date(self) -> datetime | None: ...

    @property
    def grouped_id(self) -> int | None: ...

    @property
    def message(self) -> str | None: ...


class _HistoryClient(Protocol):
    def start(self, *, phone: str) -> Awaitable[object]: ...

    def iter_messages(self, entity: str, *, reverse: bool) -> AsyncIterator[_HistoryMessage]: ...

    def download_media(
        self, message: _HistoryMessage, *, file: type[bytes]
    ) -> Awaitable[bytes | None]: ...

    def disconnect(self) -> Awaitable[None]: ...


type _HistoryClientFactory = Callable[[str, int, str], _HistoryClient]


@dataclass(frozen=True, slots=True)
class HistoryImportSummary:
    imported: int
    skipped: int
    failed: int


def _new_client(session_path: str, api_id: int, api_hash: str) -> _HistoryClient:
    raw_client = TelegramClient(session_path, api_id, api_hash)
    return cast(_HistoryClient, cast(object, raw_client))


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _is_image(message: _HistoryMessage) -> bool:
    if message.photo is not None:
        return True
    file = message.file
    return message.document is not None and bool(
        file is not None and file.mime_type is not None and file.mime_type.startswith("image/")
    )


class TelethonHistoryImporter:
    def __init__(
        self,
        services: Services,
        config: TelegramHistoryConfig,
        client_factory: _HistoryClientFactory = _new_client,
    ) -> None:
        config.validate()
        self._services = services
        self._config = config
        self._client_factory = client_factory

    async def run(self) -> HistoryImportSummary:
        self._config.session_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._config.session_path.parent.chmod(0o700)
        client = self._client_factory(
            str(self._config.session_path),
            cast(int, self._config.api_id),
            cast(str, self._config.api_hash),
        )
        await client.start(phone=cast(str, self._config.phone))
        imported = 0
        skipped = 0
        failed = 0
        try:
            async for message in client.iter_messages(self._config.channel, reverse=True):
                if not _is_image(message):
                    continue
                if message.chat_id is None:
                    failed += 1
                    error = RuntimeError(f"Channel message {message.id} has no stable chat ID")
                    self._services.catalog.record_history_import_failure(
                        f"history:unknown:{message.id}", 0, message.id, error
                    )
                    print(f"Failed channel message {message.id}: {error}")
                    continue
                chat_id = message.chat_id
                source_key = f"history:{chat_id}:{message.id}"
                if self._services.catalog.find_ingestion(source_key) is not None:
                    self._services.catalog.clear_history_import_failure(source_key)
                    skipped += 1
                    continue
                try:
                    downloaded = await client.download_media(message, file=bytes)
                    if downloaded is None:
                        raise RuntimeError(
                            f"Could not download image from channel message {message.id}"
                        )
                    file = message.file
                    file_name = file.name if file is not None else None
                    mime_type = file.mime_type if file is not None else "image/jpeg"
                    file_size = file.size if file is not None else len(downloaded)
                    telegram_file = TelegramFile(
                        file_name=file_name,
                        mime_type=mime_type,
                        file_size=file_size,
                    )
                    history_message = TelegramMessageRef(
                        role=TelegramMessageRole.HISTORY,
                        chat_id=chat_id,
                        message_id=message.id,
                        sent_at=_iso(message.date),
                        edited_at=_iso(message.edit_date),
                        media_group_id=(
                            str(message.grouped_id) if message.grouped_id is not None else None
                        ),
                        caption=message.message,
                        file=telegram_file,
                    )
                    channel_post = TelegramMessageRef(
                        role=TelegramMessageRole.CHANNEL_POST,
                        chat_id=chat_id,
                        message_id=message.id,
                        sent_at=_iso(message.date),
                        media_group_id=history_message.media_group_id,
                        caption=history_message.caption,
                        file=telegram_file,
                    )
                    await self._services.ingestion.ingest(
                        IngestRequest(
                            data=downloaded,
                            filename=file_name or f"channel-{message.id}.jpg",
                            declared_media_type=mime_type,
                            source_kind=SourceKind.CHANNEL_IMPORT,
                            source_key=source_key,
                            source_message=history_message,
                            assume_sign=True,
                            initial_sign_status=ReviewStatus.PUBLISHED,
                            existing_channel_message=channel_post,
                        )
                    )
                except Exception as error:
                    failed += 1
                    self._services.catalog.record_history_import_failure(
                        source_key, chat_id, message.id, error
                    )
                    print(f"Failed channel message {message.id}: {type(error).__name__}: {error}")
                    continue
                self._services.catalog.clear_history_import_failure(source_key)
                imported += 1
                print(f"Imported channel message {message.id} ({imported} new, {skipped} skipped)")
        finally:
            await client.disconnect()
        return HistoryImportSummary(imported=imported, skipped=skipped, failed=failed)
