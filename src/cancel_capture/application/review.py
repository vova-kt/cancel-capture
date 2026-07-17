from __future__ import annotations

from cancel_capture.models import PublishedMessage, ReviewCandidate, ReviewStatus
from cancel_capture.ports import CatalogRepository, ChannelPublisher


class ReviewService:
    def __init__(self, catalog: CatalogRepository) -> None:
        self._catalog = catalog

    async def approve(
        self,
        item_id: str,
        review_token: str,
        expected_status: ReviewStatus,
        actor_user_id: int,
        publisher: ChannelPublisher,
    ) -> PublishedMessage:
        candidate = self._catalog.claim_publish(
            item_id, review_token, expected_status, actor_user_id
        )
        try:
            published = await publisher.publish(candidate)
        except Exception as error:
            try:
                self._catalog.fail_publish(item_id, actor_user_id, str(error))
            except Exception as recovery_error:
                error.add_note(f"Could not persist the failed publish state: {recovery_error}")
            raise
        try:
            self._catalog.complete_publish(item_id, actor_user_id, published)
        except Exception as error:
            outcome = (
                "Telegram returned a successful channel post "
                f"({published.chat_id}:{published.message_id}), but saving that result failed: "
                f"{error}. Check the channel before retrying."
            )
            try:
                self._catalog.fail_publish(item_id, actor_user_id, outcome)
            except Exception as recovery_error:
                error.add_note(f"Could not persist the uncertain publish state: {recovery_error}")
            raise
        return published

    def reject(self, item_id: str, review_token: str, actor_user_id: int) -> ReviewCandidate:
        return self._catalog.reject(item_id, review_token, actor_user_id)

    def reconcile(self, item_id: str, actor_user_id: int, published: PublishedMessage) -> None:
        self._catalog.reconcile_publish(item_id, actor_user_id, published)
