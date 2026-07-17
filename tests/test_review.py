from dataclasses import dataclass

import pytest

from cancel_capture.adapters.sqlite_catalog import SQLiteCatalog
from cancel_capture.application.review import ReviewService
from cancel_capture.errors import ReviewConflictError
from cancel_capture.models import (
    BoundingBox,
    IngestRequest,
    PhotoObservation,
    PublishedMessage,
    ReviewCandidate,
    ReviewStatus,
    SignObservation,
    SourceKind,
)
from tests.fakes import build_stack, sample_jpeg


@dataclass
class FakePublisher:
    should_fail: bool = False
    calls: int = 0

    async def publish(self, candidate: ReviewCandidate) -> PublishedMessage:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("ambiguous network failure")
        return PublishedMessage(chat_id=-1001, message_id=99, sent_at="2026-07-17T10:00:00Z")


class CompletionFailCatalog(SQLiteCatalog):
    def complete_publish(self, item_id: str, actor_user_id: int, message: PublishedMessage) -> None:
        del item_id, actor_user_id, message
        raise RuntimeError("disk became unavailable")


async def _candidate(stack, source_key: str) -> str:
    result = await stack.ingestion.ingest(
        IngestRequest(
            data=sample_jpeg(),
            filename="review.jpg",
            declared_media_type="image/jpeg",
            source_kind=SourceKind.BOT,
            source_key=source_key,
        )
    )
    return result.signs[0].item_id


def _observation() -> PhotoObservation:
    return PhotoObservation(
        "A street",
        (
            SignObservation(
                ordinal=0,
                box=BoundingBox(0.1, 0.1, 0.5, 0.7),
                confidence=0.9,
                factual_summary="A bicycle is prohibited",
            ),
        ),
    )


async def test_publish_claim_is_atomic_and_cannot_double_post(tmp_path) -> None:
    stack = build_stack(tmp_path, _observation())
    item_id = await _candidate(stack, "bot:1:approve")
    publisher = FakePublisher()
    token = stack.catalog.get_candidate(item_id).review_token

    published = await stack.review.approve(item_id, token, ReviewStatus.PENDING, 123, publisher)

    assert published.message_id == 99
    assert publisher.calls == 1
    assert stack.catalog.get_candidate(item_id).status is ReviewStatus.PUBLISHED
    with pytest.raises(ReviewConflictError):
        await stack.review.approve(item_id, token, ReviewStatus.PENDING, 123, publisher)
    assert publisher.calls == 1


async def test_failed_publish_requires_explicit_retry(tmp_path) -> None:
    stack = build_stack(tmp_path, _observation())
    item_id = await _candidate(stack, "bot:1:failure")
    pending = stack.catalog.get_candidate(item_id)

    with pytest.raises(RuntimeError, match="ambiguous"):
        await stack.review.approve(
            item_id,
            pending.review_token,
            ReviewStatus.PENDING,
            123,
            FakePublisher(should_fail=True),
        )
    failed = stack.catalog.get_candidate(item_id)
    assert failed.status is ReviewStatus.FAILED
    assert failed.last_error == "ambiguous network failure"
    assert failed.review_token != pending.review_token

    await stack.review.approve(
        item_id,
        failed.review_token,
        ReviewStatus.FAILED,
        123,
        FakePublisher(),
    )
    assert stack.catalog.get_candidate(item_id).status is ReviewStatus.PUBLISHED


async def test_replayed_retry_token_cannot_start_another_publish(tmp_path) -> None:
    stack = build_stack(tmp_path, _observation())
    item_id = await _candidate(stack, "bot:1:replayed-retry")
    pending = stack.catalog.get_candidate(item_id)
    with pytest.raises(RuntimeError):
        await stack.review.approve(
            item_id,
            pending.review_token,
            ReviewStatus.PENDING,
            123,
            FakePublisher(should_fail=True),
        )
    first_failure = stack.catalog.get_candidate(item_id)
    with pytest.raises(RuntimeError):
        await stack.review.approve(
            item_id,
            first_failure.review_token,
            ReviewStatus.FAILED,
            123,
            FakePublisher(should_fail=True),
        )

    replayed = FakePublisher()
    with pytest.raises(ReviewConflictError):
        await stack.review.approve(
            item_id,
            first_failure.review_token,
            ReviewStatus.FAILED,
            123,
            replayed,
        )
    assert replayed.calls == 0


async def test_interrupted_publish_is_recovered_as_uncertain(tmp_path) -> None:
    stack = build_stack(tmp_path, _observation())
    item_id = await _candidate(stack, "bot:1:interrupted")
    pending = stack.catalog.get_candidate(item_id)
    stack.catalog.claim_publish(item_id, pending.review_token, ReviewStatus.PENDING, 123)

    assert stack.catalog.recover_interrupted_publishes() == 1
    assert stack.catalog.recover_interrupted_publishes() == 0
    candidate = stack.catalog.get_candidate(item_id)
    assert candidate.status is ReviewStatus.FAILED
    assert candidate.last_error is not None
    assert "outcome is uncertain" in candidate.last_error
    assert candidate.review_token != pending.review_token


async def test_successful_telegram_post_with_db_failure_becomes_uncertain(tmp_path) -> None:
    stack = build_stack(tmp_path, _observation())
    item_id = await _candidate(stack, "bot:1:completion-failure")
    catalog = CompletionFailCatalog(tmp_path / "data" / "catalog.sqlite3")
    catalog.initialize()
    pending = catalog.get_candidate(item_id)

    with pytest.raises(RuntimeError, match="disk became unavailable"):
        await ReviewService(catalog).approve(
            item_id,
            pending.review_token,
            ReviewStatus.PENDING,
            123,
            FakePublisher(),
        )

    candidate = catalog.get_candidate(item_id)
    assert candidate.status is ReviewStatus.FAILED
    assert candidate.last_error is not None
    assert "-1001:99" in candidate.last_error
    assert "Check the channel" in candidate.last_error


async def test_uncertain_publish_can_link_verified_existing_message(tmp_path) -> None:
    stack = build_stack(tmp_path, _observation())
    item_id = await _candidate(stack, "bot:1:reconcile")
    pending = stack.catalog.get_candidate(item_id)
    with pytest.raises(RuntimeError):
        await stack.review.approve(
            item_id,
            pending.review_token,
            ReviewStatus.PENDING,
            123,
            FakePublisher(should_fail=True),
        )

    stack.review.reconcile(
        item_id,
        123,
        PublishedMessage(chat_id=-1001, message_id=77, sent_at=None),
    )

    assert stack.catalog.get_candidate(item_id).status is ReviewStatus.PUBLISHED
    with pytest.raises(ReviewConflictError):
        stack.review.reconcile(
            item_id,
            123,
            PublishedMessage(chat_id=-1001, message_id=77, sent_at=None),
        )


async def test_rejection_retains_crop_and_is_terminal(tmp_path) -> None:
    stack = build_stack(tmp_path, _observation())
    item_id = await _candidate(stack, "bot:1:reject")
    pending = stack.catalog.get_candidate(item_id)
    path = stack.assets.resolve(pending.crop_relative_path)

    rejected = stack.review.reject(item_id, pending.review_token, 123)

    assert rejected.status is ReviewStatus.REJECTED
    assert path.exists()
    with pytest.raises(ReviewConflictError):
        stack.review.reject(item_id, pending.review_token, 123)
