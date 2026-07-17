from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

from telegram import InputFile
from telegram.constants import ChatType

from cancel_capture.adapters.telegram_bot import BotHandlers, TelegramChannelPublisher
from cancel_capture.config import AppConfig
from cancel_capture.container import Services
from cancel_capture.models import (
    BoundingBox,
    IngestRequest,
    PhotoObservation,
    SignObservation,
    SourceKind,
)
from tests.fakes import build_stack, sample_jpeg


class FakeTelegramBot:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    async def send_photo(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            chat_id=-100123,
            message_id=42,
            date=datetime(2026, 7, 17, tzinfo=UTC),
            photo=[],
        )


def _config(tmp_path) -> AppConfig:
    return AppConfig.from_env(
        {
            "DATA_DIR": str(tmp_path / "data"),
            "OPENAI_API_KEY": "unused",
            "TELEGRAM_BOT_TOKEN": "123:test",
            "TELEGRAM_OWNER_USER_ID": "77",
        },
        load_dotenv_file=False,
    )


def _observation() -> PhotoObservation:
    return PhotoObservation(
        "A street",
        (
            SignObservation(
                ordinal=0,
                box=BoundingBox(0.1, 0.1, 0.6, 0.8),
                confidence=0.9,
                factual_summary="A bicycle is prohibited",
            ),
        ),
    )


async def test_channel_publisher_uses_ephemeral_telegram_rendition(tmp_path) -> None:
    stack = build_stack(tmp_path, _observation())
    result = await stack.ingestion.ingest(
        IngestRequest(
            data=sample_jpeg(),
            filename="source.jpg",
            declared_media_type="image/jpeg",
            source_kind=SourceKind.BOT,
            source_key="bot:77:publisher",
        )
    )
    candidate = stack.catalog.get_candidate(result.signs[0].item_id)
    archive_path = stack.assets.resolve(candidate.crop_relative_path)
    archive_bytes = archive_path.read_bytes()
    fake_bot = FakeTelegramBot()

    published = await TelegramChannelPublisher(
        cast(object, fake_bot),
        "@cancel_capture",
        cast(Services, cast(object, stack)),
    ).publish(candidate)

    assert published.message_id == 42
    assert fake_bot.request is not None
    upload = fake_bot.request["photo"]
    assert isinstance(upload, InputFile)
    assert len(upload.input_file_content) < 10_000_000
    assert archive_path.read_bytes() == archive_bytes


def test_bot_authorization_requires_numeric_owner_and_private_chat(tmp_path) -> None:
    stack = build_stack(tmp_path, _observation())
    config = _config(tmp_path)
    handlers = BotHandlers(cast(Services, cast(object, stack)), config.bot)

    assert handlers._authorized(
        cast(
            object,
            SimpleNamespace(
                effective_user=SimpleNamespace(id=77),
                effective_chat=SimpleNamespace(type=ChatType.PRIVATE),
            ),
        )
    )
    assert not handlers._authorized(
        cast(
            object,
            SimpleNamespace(
                effective_user=SimpleNamespace(id=78),
                effective_chat=SimpleNamespace(type=ChatType.PRIVATE),
            ),
        )
    )
    assert not handlers._authorized(
        cast(
            object,
            SimpleNamespace(
                effective_user=SimpleNamespace(id=77),
                effective_chat=SimpleNamespace(type=ChatType.GROUP),
            ),
        )
    )
