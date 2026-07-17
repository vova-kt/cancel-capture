import sqlite3
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

from cancel_capture.adapters.telethon_history import TelethonHistoryImporter
from cancel_capture.config import AppConfig
from cancel_capture.container import Services
from cancel_capture.models import PhotoObservation
from tests.fakes import build_stack, sample_jpeg


@dataclass(frozen=True)
class FakeHistoryFile:
    name: str | None = "sign.jpg"
    mime_type: str | None = "image/jpeg"
    size: int | None = None


@dataclass(frozen=True)
class FakeHistoryMessage:
    id: int
    chat_id: int | None
    payload: bytes | None
    photo: object | None = object()
    document: object | None = None
    file: FakeHistoryFile | None = FakeHistoryFile()
    date: datetime | None = datetime(2026, 7, 17, tzinfo=UTC)
    edit_date: datetime | None = None
    grouped_id: int | None = None
    message: str | None = None


class FakeHistoryClient:
    def __init__(self, messages: tuple[FakeHistoryMessage, ...]) -> None:
        self.messages = messages
        self.started = False
        self.disconnected = False

    async def start(self, *, phone: str) -> object:
        assert phone == "+491234"
        self.started = True
        return self

    async def iter_messages(
        self, entity: str, *, reverse: bool
    ) -> AsyncIterator[FakeHistoryMessage]:
        assert entity == "@cancel_capture"
        assert reverse is True
        for message in self.messages:
            yield message

    async def download_media(
        self, message: FakeHistoryMessage, *, file: type[bytes]
    ) -> bytes | None:
        assert file is bytes
        return message.payload

    async def disconnect(self) -> None:
        self.disconnected = True


async def test_importer_records_bad_message_and_continues_then_reruns(tmp_path) -> None:
    stack = build_stack(tmp_path, PhotoObservation("An existing prohibition sign", ()))
    config = AppConfig.from_env(
        {
            "DATA_DIR": str(tmp_path / "data"),
            "OPENAI_API_KEY": "unused",
            "TG_API_ID": "123",
            "TG_API_HASH": "hash",
            "TG_PHONE": "+491234",
        },
        load_dotenv_file=False,
    )
    services = Services(
        config=config,
        assets=stack.assets,
        catalog=stack.catalog,
        ingestion=stack.ingestion,
        review=stack.review,
        search=stack.search,
    )
    client = FakeHistoryClient(
        (
            FakeHistoryMessage(id=1, chat_id=-100123, payload=b"not an image"),
            FakeHistoryMessage(id=2, chat_id=-100123, payload=sample_jpeg()),
        )
    )
    importer = TelethonHistoryImporter(services, config.history, lambda *_args: client)

    first = await importer.run()
    second = await importer.run()

    assert first.imported == 1
    assert first.skipped == 0
    assert first.failed == 1
    assert second.imported == 0
    assert second.skipped == 1
    assert second.failed == 1
    assert client.started is True
    assert client.disconnected is True
    assert stack.catalog.find_ingestion("history:-100123:2") is not None
    with sqlite3.connect(tmp_path / "data" / "catalog.sqlite3") as connection:
        failure = connection.execute(
            """
            SELECT attempts, error_type FROM history_import_failures
            WHERE source_key = 'history:-100123:1'
            """
        ).fetchone()
    assert failure is not None
    assert failure[0] == 2
    assert failure[1]
