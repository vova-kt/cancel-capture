from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC
from typing import Any, cast

from telegram import (
    Bot,
    ChatMemberAdministrator,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    Update,
)
from telegram.constants import ChatType
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackContext,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ExtBot,
    JobQueue,
    MessageHandler,
    filters,
)

from cancel_capture.adapters.telegram_photo import render_telegram_photo
from cancel_capture.config import TelegramBotConfig
from cancel_capture.container import Services
from cancel_capture.errors import ReviewConflictError
from cancel_capture.models import (
    ImageMetadata,
    IngestRequest,
    PublishedMessage,
    ReviewCandidate,
    ReviewStatus,
    SourceKind,
    TelegramFile,
    TelegramMessageRef,
    TelegramMessageRole,
)

type _Data = dict[Any, Any]
type _CallbackContext = CallbackContext[ExtBot[None], _Data, _Data, _Data]
type _Application = Application[
    ExtBot[None],
    _CallbackContext,
    _Data,
    _Data,
    _Data,
    JobQueue[_CallbackContext],
]

_CALLBACK = re.compile(r"^(publish|retry|reject):([0-9a-f]{32}):([0-9a-f]{16})$")
_ITEM_ID = re.compile(r"^[0-9a-f]{32}$")


def parse_callback(data: str) -> tuple[str, str, str] | None:
    match = _CALLBACK.fullmatch(data)
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def _review_keyboard(
    item_id: str, review_token: str, *, retry: bool = False
) -> InlineKeyboardMarkup:
    publish_label = "🔄 Retry publish" if retry else "✅ Publish"
    publish_action = "retry" if retry else "publish"
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    publish_label,
                    callback_data=f"{publish_action}:{item_id}:{review_token}",
                ),
                InlineKeyboardButton("❌ Reject", callback_data=f"reject:{item_id}:{review_token}"),
            ]
        ]
    )


def _metadata_summary(metadata: ImageMetadata) -> str:
    fields: list[str] = [f"Extractor: {metadata.extractor}"]
    fields.append(f"Captured: {metadata.captured_at or 'not available'}")
    if metadata.latitude is not None and metadata.longitude is not None:
        fields.append(f"GPS (private archive): {metadata.latitude:.6f}, {metadata.longitude:.6f}")
    else:
        fields.append("GPS: not available")
    camera = " ".join(
        value for value in (metadata.camera_make, metadata.camera_model) if value is not None
    )
    fields.append(f"Camera: {camera or 'not available'}")
    if metadata.lens_model is not None:
        fields.append(f"Lens: {metadata.lens_model}")
    try:
        raw = cast(object, json.loads(metadata.raw_json))
        raw_count = len(cast(dict[object, object], raw)) if isinstance(raw, dict) else 0
    except json.JSONDecodeError:
        raw_count = 0
    fields.append(f"Raw metadata fields saved: {raw_count}")
    return "\n".join(fields)


def _card_text(candidate: ReviewCandidate) -> str:
    visible = ", ".join(candidate.visible_text) or "none"
    topics_en = ", ".join(candidate.topics_en) or "none"
    topics_ru = ", ".join(candidate.topics_ru) or "нет"
    text = (
        f"Archive ID: {candidate.item_id}\n\n"
        f"PHOTO / ФОТО\nEN: {candidate.photo_description.en}\n"
        f"RU: {candidate.photo_description.ru}\n\n"
        f"SIGN {candidate.ordinal + 1} / ЗНАК {candidate.ordinal + 1}\n"
        f"EN: {candidate.sign_description.en}\n"
        f"RU: {candidate.sign_description.ru}\n\n"
        f"Visible text: {visible}\nTopics: {topics_en}\nТемы: {topics_ru}\n\n"
        f"METADATA / МЕТАДАННЫЕ\n{_metadata_summary(candidate.metadata)}"
    )
    if len(text) <= 4000:
        return text
    return f"{text[:3950]}\n… Full descriptions and metadata are saved in SQLite."


class TelegramChannelPublisher:
    def __init__(self, bot: Bot, channel: str, services: Services) -> None:
        self._bot = bot
        self._channel = channel
        self._services = services

    async def publish(self, candidate: ReviewCandidate) -> PublishedMessage:
        path = self._services.assets.resolve(candidate.crop_relative_path)
        rendition = render_telegram_photo(path)
        message = await self._bot.send_photo(
            chat_id=self._channel,
            photo=InputFile(rendition, filename=f"sign-{candidate.item_id}.jpg"),
        )
        largest = message.photo[-1] if message.photo else None
        return PublishedMessage(
            chat_id=message.chat_id,
            message_id=message.message_id,
            sent_at=message.date.astimezone(UTC).isoformat(),
            file=(
                TelegramFile(
                    file_id=largest.file_id,
                    file_unique_id=largest.file_unique_id,
                    file_size=largest.file_size,
                )
                if largest is not None
                else None
            ),
        )


class BotHandlers:
    def __init__(self, services: Services, config: TelegramBotConfig) -> None:
        config.validate()
        self._services = services
        self._config = config
        self._owner_user_id = cast(int, config.owner_user_id)

    def register(
        self,
        application: _Application,
    ) -> None:
        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("status", self.status))
        application.add_handler(CommandHandler("markpublished", self.mark_published))
        application.add_handler(CallbackQueryHandler(self.callback, pattern=_CALLBACK.pattern))
        application.add_handler(MessageHandler(filters.PHOTO, self.compressed_photo))
        application.add_handler(MessageHandler(filters.Document.ALL, self.document))

    def _authorized(self, update: Update) -> bool:
        user = update.effective_user
        chat = update.effective_chat
        return (
            user is not None
            and user.id == self._owner_user_id
            and chat is not None
            and chat.type == ChatType.PRIVATE
        )

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not self._authorized(update) or update.effective_message is None:
            return
        await update.effective_message.reply_text(
            "Send an original image as a file/document. I will archive it, detect every "
            "prohibition sign, and send one bilingual review card per crop."
        )

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not self._authorized(update) or update.effective_message is None:
            return
        pending = self._services.catalog.list_candidates(ReviewStatus.PENDING)
        failed = self._services.catalog.list_candidates(ReviewStatus.FAILED)
        await update.effective_message.reply_text(
            f"Pending review: {len(pending)}\nFailed publish attempts: {len(failed)}"
        )

    async def compressed_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        if not self._authorized(update) or update.effective_message is None:
            return
        await update.effective_message.reply_text(
            "Please resend this image as a file/document so Telegram does not compress it or strip "
            "the remaining metadata."
        )

    async def mark_published(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update) or update.effective_message is None:
            return
        arguments = context.args or []
        if len(arguments) != 2 or _ITEM_ID.fullmatch(arguments[0]) is None:
            await update.effective_message.reply_text(
                "Usage: /markpublished <32-character archive ID> <channel message ID>"
            )
            return
        try:
            channel_message_id = int(arguments[1])
        except ValueError:
            await update.effective_message.reply_text("Channel message ID must be an integer.")
            return
        if channel_message_id <= 0:
            await update.effective_message.reply_text("Channel message ID must be positive.")
            return

        item_id = arguments[0]
        candidate = self._services.catalog.get_candidate(item_id)
        if candidate.status is not ReviewStatus.FAILED:
            await update.effective_message.reply_text(
                "Only a failed or uncertain publish can be reconciled."
            )
            return
        channel = await context.bot.get_chat(self._config.channel)
        verified = await context.bot.forward_message(
            chat_id=update.effective_message.chat_id,
            from_chat_id=channel.id,
            message_id=channel_message_id,
        )
        verified_document = verified.document
        if not verified.photo and not (
            verified_document is not None
            and verified_document.mime_type is not None
            and verified_document.mime_type.startswith("image/")
        ):
            await update.effective_message.reply_text(
                "That channel message is not an image, so it was not linked."
            )
            return
        self._services.review.reconcile(
            item_id,
            self._owner_user_id,
            PublishedMessage(
                chat_id=channel.id,
                message_id=channel_message_id,
                sent_at=None,
            ),
        )
        await update.effective_message.reply_text(
            "The forwarded channel image was verified and linked to this archive item as published."
        )

    async def document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._authorized(update):
            return
        message = update.effective_message
        if message is None or message.document is None:
            return
        document = message.document
        if document.file_size is not None and document.file_size > self._config.max_download_bytes:
            await message.reply_text(
                f"That file is {document.file_size} bytes. This Bot API endpoint is configured for "
                f"at most {self._config.max_download_bytes} bytes."
            )
            return

        progress = await message.reply_text("Archiving and analyzing the original file…")
        telegram_file = await context.bot.get_file(document.file_id)
        downloaded = bytes(await telegram_file.download_as_bytearray())
        if len(downloaded) > self._config.max_download_bytes:
            await progress.edit_text(
                f"The downloaded file is {len(downloaded)} bytes, above the configured "
                f"{self._config.max_download_bytes}-byte Bot API limit. It was not archived."
            )
            return
        source_message = TelegramMessageRef(
            role=TelegramMessageRole.INBOUND,
            chat_id=message.chat_id,
            message_id=message.message_id,
            sent_at=message.date.astimezone(UTC).isoformat(),
            media_group_id=message.media_group_id,
            caption=message.caption,
            file=TelegramFile(
                file_id=document.file_id,
                file_unique_id=document.file_unique_id,
                file_name=document.file_name,
                mime_type=document.mime_type,
                file_size=document.file_size,
            ),
        )
        try:
            result = await self._services.ingestion.ingest(
                IngestRequest(
                    data=downloaded,
                    filename=document.file_name,
                    declared_media_type=document.mime_type,
                    source_kind=SourceKind.BOT,
                    source_key=f"bot:{message.chat_id}:{message.message_id}",
                    source_message=source_message,
                )
            )
        except Exception as error:
            await progress.edit_text(f"Analysis failed: {error}")
            raise

        if not result.signs:
            await progress.edit_text(
                "The original and its full-photo description were archived, but no qualifying "
                "round red crossed-out signs were detected."
            )
            return
        candidates = tuple(
            self._services.catalog.get_candidate(sign.item_id) for sign in result.signs
        )
        deliverable = tuple(
            candidate
            for candidate in candidates
            if candidate.status in (ReviewStatus.PENDING, ReviewStatus.FAILED)
            and not self._services.catalog.has_preview(candidate.item_id)
        )
        if not deliverable:
            await progress.edit_text(
                "This Telegram upload was already processed; no new review cards were created."
            )
            return
        await progress.edit_text(
            f"Detected {len(result.signs)} sign(s); sending {len(deliverable)} new review card(s)."
        )
        for index, candidate in enumerate(deliverable):
            crop_path = self._services.assets.resolve(candidate.crop_relative_path)
            rendition = render_telegram_photo(crop_path)
            preview = await message.reply_photo(
                photo=InputFile(rendition, filename=f"sign-{candidate.item_id}.jpg"),
                caption=f"Crop {candidate.ordinal + 1} / Кадр {candidate.ordinal + 1}",
            )
            card = await message.reply_text(
                _card_text(candidate),
                reply_to_message_id=preview.message_id,
                reply_markup=_review_keyboard(
                    candidate.item_id,
                    candidate.review_token,
                    retry=candidate.status is ReviewStatus.FAILED,
                ),
            )
            self._services.catalog.record_preview(
                candidate.item_id,
                TelegramMessageRef(
                    role=TelegramMessageRole.PREVIEW,
                    chat_id=card.chat_id,
                    message_id=card.message_id,
                    sent_at=card.date.astimezone(UTC).isoformat(),
                ),
            )
            if index + 1 < len(deliverable):
                await asyncio.sleep(1.0)

    async def callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None:
            return
        await query.answer()
        if not self._authorized(update) or query.data is None:
            return
        parsed = parse_callback(query.data)
        if parsed is None:
            return
        action, item_id, review_token = parsed
        try:
            if action == "reject":
                self._services.review.reject(item_id, review_token, self._owner_user_id)
                await query.edit_message_reply_markup(reply_markup=None)
                if update.effective_chat is not None:
                    await context.bot.send_message(
                        update.effective_chat.id,
                        f"Rejected sign {item_id[:8]}. Original retained.",
                    )
                return

            publisher = TelegramChannelPublisher(context.bot, self._config.channel, self._services)
            expected_status = ReviewStatus.FAILED if action == "retry" else ReviewStatus.PENDING
            published = await self._services.review.approve(
                item_id,
                review_token,
                expected_status,
                self._owner_user_id,
                publisher,
            )
            await query.edit_message_reply_markup(reply_markup=None)
            if update.effective_chat is not None:
                await context.bot.send_message(
                    update.effective_chat.id,
                    f"Published sign {item_id[:8]} as channel message {published.message_id}.",
                )
        except ReviewConflictError as error:
            candidate = self._services.catalog.get_candidate(item_id)
            if candidate.status in (ReviewStatus.PENDING, ReviewStatus.FAILED):
                await query.edit_message_reply_markup(
                    reply_markup=_review_keyboard(
                        candidate.item_id,
                        candidate.review_token,
                        retry=candidate.status is ReviewStatus.FAILED,
                    )
                )
            if update.effective_chat is not None:
                await context.bot.send_message(
                    update.effective_chat.id,
                    f"That review button expired and did nothing. Re-check the current state "
                    f"before using the refreshed button. {error}",
                )
        except Exception as error:
            candidate = self._services.catalog.get_candidate(item_id)
            keyboard = (
                _review_keyboard(item_id, candidate.review_token, retry=True)
                if candidate.status is ReviewStatus.FAILED
                else None
            )
            await query.edit_message_reply_markup(reply_markup=keyboard)
            if update.effective_chat is not None:
                await context.bot.send_message(
                    update.effective_chat.id,
                    "Publish failed or its result is uncertain. Check the channel before using "
                    f"Retry. Error: {error}",
                )


async def verify_channel_access(
    application: _Application,
    channel: str,
) -> None:
    me = await application.bot.get_me()
    member = await application.bot.get_chat_member(channel, me.id)
    if not isinstance(member, ChatMemberAdministrator) or not member.can_post_messages:
        raise RuntimeError(
            f"Bot @{me.username} is not an administrator allowed to post in {channel}"
        )


def build_bot_application(
    services: Services,
) -> _Application:
    config = services.config.bot
    config.validate()
    builder = ApplicationBuilder().token(cast(str, config.token))
    if (config.api_base_url is None) != (config.api_file_url is None):
        raise ValueError("Both Telegram Bot API base URLs must be set together")
    if config.api_base_url is not None and config.api_file_url is not None:
        builder = builder.base_url(config.api_base_url).base_file_url(config.api_file_url)
    builder = builder.post_init(  # pyright: ignore[reportUnknownMemberType]
        lambda app: verify_channel_access(  # pyright: ignore[reportUnknownLambdaType]
            app,  # pyright: ignore[reportUnknownArgumentType]
            config.channel,
        )
    )
    application = builder.build()
    BotHandlers(services, config).register(application)
    return application


def run_bot(services: Services) -> None:
    recovered = services.catalog.recover_interrupted_publishes()
    if recovered:
        print(
            f"Recovered {recovered} interrupted publish attempt(s) as failed; "
            "check the channel before retrying."
        )
    application = build_bot_application(services)
    application.run_polling(drop_pending_updates=False, allowed_updates=Update.ALL_TYPES)
