import sqlite3
from io import BytesIO

from PIL import Image

from cancel_capture.models import (
    BoundingBox,
    IngestRequest,
    ItemKind,
    PhotoObservation,
    ProviderIdentity,
    ReviewStatus,
    SignObservation,
    SourceKind,
    TelegramMessageRef,
    TelegramMessageRole,
)
from tests.fakes import build_stack, sample_jpeg


def two_sign_observation() -> PhotoObservation:
    return PhotoObservation(
        factual_summary="A street entrance with two signs",
        signs=(
            SignObservation(
                ordinal=0,
                box=BoundingBox(0.08, 0.12, 0.45, 0.68),
                confidence=0.97,
                factual_summary="A bicycle is prohibited",
                visible_text=("NO BIKES",),
            ),
            SignObservation(
                ordinal=1,
                box=BoundingBox(0.55, 0.15, 0.92, 0.72),
                confidence=0.92,
                factual_summary="A dog is prohibited",
            ),
        ),
    )


async def test_ingestion_persists_photo_signs_metadata_and_embeddings(tmp_path) -> None:
    stack = build_stack(tmp_path, two_sign_observation())
    data = sample_jpeg()
    result = await stack.ingestion.ingest(
        IngestRequest(
            data=data,
            filename="original.jpg",
            declared_media_type="image/jpeg",
            source_kind=SourceKind.BOT,
            source_key="bot:10:20",
        )
    )

    assert stack.assets.resolve(result.original_relative_path).read_bytes() == data
    assert len(result.signs) == 2
    assert all(sign.status is ReviewStatus.PENDING for sign in result.signs)
    assert result.metadata.latitude == 52.5
    assert stack.vision.calls == 1
    assert stack.text.calls == 1
    assert stack.embeddings.calls == 1
    assert len(stack.catalog.list_search_documents(stack.embeddings.identity, 3)) == 3
    assert stack.catalog.list_search_documents(stack.embeddings.identity, 4) == ()
    assert (
        stack.catalog.list_search_documents(
            ProviderIdentity("fake", "embedding-v1", "other-deployment"), 3
        )
        == ()
    )

    first = stack.catalog.get_candidate(result.signs[0].item_id)
    assert first.visible_text == ("NO BIKES",)
    assert first.topics_ru == ("велосипед", "мобильность")
    assert stack.assets.resolve(first.crop_relative_path).is_file()
    assert stack.catalog.has_preview(first.item_id) is False
    stack.catalog.record_preview(
        first.item_id,
        TelegramMessageRef(
            role=TelegramMessageRole.PREVIEW,
            chat_id=10,
            message_id=99,
        ),
    )
    assert stack.catalog.has_preview(first.item_id) is True


async def test_ingestion_records_source_dimensions_and_full_resolution_crop(tmp_path) -> None:
    image = Image.new("RGB", (1200, 800), "white")
    source = BytesIO()
    image.save(source, format="JPEG")
    observation = PhotoObservation(
        factual_summary="An existing prohibition sign",
        signs=(
            SignObservation(
                ordinal=0,
                box=BoundingBox.full_frame(),
                confidence=0.99,
                factual_summary="A prohibition sign",
            ),
        ),
    )
    stack = build_stack(tmp_path, observation, max_analysis_side=100)

    result = await stack.ingestion.ingest(
        IngestRequest(
            data=source.getvalue(),
            filename="large.jpg",
            declared_media_type="image/jpeg",
            source_kind=SourceKind.BOT,
            source_key="bot:10:full-resolution",
        )
    )

    candidate = stack.catalog.get_candidate(result.signs[0].item_id)
    with Image.open(stack.assets.resolve(candidate.crop_relative_path)) as crop:
        assert crop.size == (1200, 800)

    with sqlite3.connect(tmp_path / "data" / "catalog.sqlite3") as connection:
        dimensions = connection.execute(
            "SELECT width, height FROM assets WHERE relative_path = ?",
            (result.original_relative_path,),
        ).fetchone()
    assert dimensions == (1200, 800)


async def test_duplicate_source_is_idempotent_before_provider_calls(tmp_path) -> None:
    stack = build_stack(tmp_path, two_sign_observation())
    request = IngestRequest(
        data=sample_jpeg(),
        filename="original.jpg",
        declared_media_type="image/jpeg",
        source_kind=SourceKind.BOT,
        source_key="bot:10:duplicate",
    )
    first = await stack.ingestion.ingest(request)
    second = await stack.ingestion.ingest(request)

    assert second.photo_item_id == first.photo_item_id
    assert second.already_existed is True
    assert stack.vision.calls == 1
    assert stack.text.calls == 1
    assert stack.embeddings.calls == 1


async def test_history_import_falls_back_to_full_frame_sign(tmp_path) -> None:
    stack = build_stack(tmp_path, PhotoObservation("An already cropped sign", ()))
    result = await stack.ingestion.ingest(
        IngestRequest(
            data=sample_jpeg(),
            filename="history.jpg",
            declared_media_type="image/jpeg",
            source_kind=SourceKind.CHANNEL_IMPORT,
            source_key="history:-100123:7",
            assume_sign=True,
            initial_sign_status=ReviewStatus.PUBLISHED,
        )
    )

    assert len(result.signs) == 1
    assert result.signs[0].status is ReviewStatus.PUBLISHED


async def test_history_post_can_be_linked_to_every_sign_in_one_image(tmp_path) -> None:
    stack = build_stack(tmp_path, two_sign_observation())
    channel_post = TelegramMessageRef(
        role=TelegramMessageRole.CHANNEL_POST,
        chat_id=-100123,
        message_id=8,
    )
    result = await stack.ingestion.ingest(
        IngestRequest(
            data=sample_jpeg(),
            filename="history.jpg",
            declared_media_type="image/jpeg",
            source_kind=SourceKind.CHANNEL_IMPORT,
            source_key="history:-100123:8",
            initial_sign_status=ReviewStatus.PUBLISHED,
            existing_channel_message=channel_post,
        )
    )

    with sqlite3.connect(tmp_path / "data" / "catalog.sqlite3") as connection:
        count = connection.execute(
            """
            SELECT count(*) FROM telegram_messages
            WHERE chat_id = -100123 AND message_id = 8 AND role = 'channel_post'
            """
        ).fetchone()
    assert len(result.signs) == 2
    assert count == (2,)


async def test_bilingual_semantic_and_lexical_search(tmp_path) -> None:
    stack = build_stack(tmp_path, two_sign_observation())
    await stack.ingestion.ingest(
        IngestRequest(
            data=sample_jpeg(),
            filename="search.jpg",
            declared_media_type="image/jpeg",
            source_kind=SourceKind.STREAMLIT,
            source_key="streamlit:search",
        )
    )

    english = await stack.search.search("bicycle restrictions", kind=ItemKind.SIGN)
    russian = await stack.search.search("запрет велосипедов", kind=ItemKind.SIGN)

    assert english[0].description.en.startswith("Sign: A bicycle")
    assert russian[0].description.en.startswith("Sign: A bicycle")
    assert english[0].score > english[1].score
    assert stack.catalog.lexical_matches("велосипед", ItemKind.SIGN.value)
