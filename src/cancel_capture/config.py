from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from dotenv import load_dotenv

from cancel_capture.errors import ConfigurationError


def _optional(environment: Mapping[str, str], name: str) -> str | None:
    value = environment.get(name, "").strip()
    return value or None


def _positive_int(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer") from error
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return value


def _identity_namespace(environment: Mapping[str, str], role: str, base_url: str | None) -> str:
    configured = _optional(environment, f"{role}_IDENTITY_NAMESPACE")
    if configured is not None:
        return configured
    if base_url is None:
        return "default"
    digest = sha256(base_url.encode("utf-8")).hexdigest()[:16]
    return f"endpoint-{digest}"


@dataclass(frozen=True, slots=True)
class StorageConfig:
    data_dir: Path
    sqlite_path: Path
    max_upload_bytes: int
    max_image_pixels: int
    max_analysis_side: int = 4096


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    provider: str
    api_key: str | None
    base_url: str | None
    model: str
    identity_namespace: str
    dimensions: int | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ConfigurationError("Provider name cannot be empty")
        if not self.model.strip():
            raise ConfigurationError("Provider model cannot be empty")
        if not self.identity_namespace.strip():
            raise ConfigurationError("Provider identity namespace cannot be empty")

    def require_api_key(self) -> str:
        if self.api_key is None:
            raise ConfigurationError(
                f"No API key configured for {self.provider!r} model {self.model!r}"
            )
        return self.api_key


@dataclass(frozen=True, slots=True)
class TelegramBotConfig:
    token: str | None
    owner_user_id: int | None
    channel: str
    api_base_url: str | None
    api_file_url: str | None
    max_download_bytes: int

    def validate(self) -> None:
        missing: list[str] = []
        if self.token is None:
            missing.append("TELEGRAM_BOT_TOKEN")
        if self.owner_user_id is None:
            missing.append("TELEGRAM_OWNER_USER_ID")
        if missing:
            raise ConfigurationError(f"Missing bot configuration: {', '.join(missing)}")


@dataclass(frozen=True, slots=True)
class TelegramHistoryConfig:
    api_id: int | None
    api_hash: str | None
    phone: str | None
    session_path: Path
    channel: str

    def validate(self) -> None:
        missing: list[str] = []
        if self.api_id is None:
            missing.append("TG_API_ID")
        if self.api_hash is None:
            missing.append("TG_API_HASH")
        if self.phone is None:
            missing.append("TG_PHONE")
        if missing:
            raise ConfigurationError(f"Missing history-import configuration: {', '.join(missing)}")


@dataclass(frozen=True, slots=True)
class AppConfig:
    storage: StorageConfig
    vision: ProviderConfig
    text: ProviderConfig
    embedding: ProviderConfig
    narrative: ProviderConfig
    bot: TelegramBotConfig
    history: TelegramHistoryConfig

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        load_dotenv_file: bool = True,
    ) -> AppConfig:
        if environment is None:
            if load_dotenv_file:
                load_dotenv()
            environment = os.environ

        data_dir = Path(environment.get("DATA_DIR", "./data")).expanduser()
        sqlite_path = Path(
            environment.get("SQLITE_PATH", str(data_dir / "catalog.sqlite3"))
        ).expanduser()
        openai_key = _optional(environment, "OPENAI_API_KEY")

        dimensions_raw = _optional(environment, "EMBEDDING_DIMENSIONS")
        dimensions: int | None = None
        if dimensions_raw is not None:
            try:
                dimensions = int(dimensions_raw)
            except ValueError as error:
                raise ConfigurationError("EMBEDDING_DIMENSIONS must be an integer") from error
            if dimensions <= 0:
                raise ConfigurationError("EMBEDDING_DIMENSIONS must be positive")

        owner_raw = _optional(environment, "TELEGRAM_OWNER_USER_ID")
        owner_user_id: int | None = None
        if owner_raw is not None:
            try:
                owner_user_id = int(owner_raw)
            except ValueError as error:
                raise ConfigurationError("TELEGRAM_OWNER_USER_ID must be an integer") from error

        api_id_raw = _optional(environment, "TG_API_ID")
        api_id: int | None = None
        if api_id_raw is not None:
            try:
                api_id = int(api_id_raw)
            except ValueError as error:
                raise ConfigurationError("TG_API_ID must be an integer") from error

        channel = environment.get("TELEGRAM_CHANNEL", "@cancel_capture").strip()
        if not channel:
            raise ConfigurationError("TELEGRAM_CHANNEL cannot be empty")

        vision_base_url = _optional(environment, "VISION_BASE_URL")
        text_base_url = _optional(environment, "TEXT_BASE_URL")
        embedding_base_url = _optional(environment, "EMBEDDING_BASE_URL")
        narrative_base_url = _optional(environment, "NARRATIVE_BASE_URL")

        return cls(
            storage=StorageConfig(
                data_dir=data_dir,
                sqlite_path=sqlite_path,
                max_upload_bytes=_positive_int(environment, "MAX_UPLOAD_BYTES", 100 * 1024 * 1024),
                max_image_pixels=_positive_int(environment, "MAX_IMAGE_PIXELS", 80_000_000),
            ),
            vision=ProviderConfig(
                provider=environment.get("VISION_PROVIDER", "openai").strip().lower(),
                api_key=_optional(environment, "VISION_API_KEY") or openai_key,
                base_url=vision_base_url,
                model=environment.get("VISION_MODEL", "gpt-5.6-terra").strip(),
                identity_namespace=_identity_namespace(environment, "VISION", vision_base_url),
            ),
            text=ProviderConfig(
                provider=environment.get("TEXT_PROVIDER", "openai").strip().lower(),
                api_key=_optional(environment, "TEXT_API_KEY") or openai_key,
                base_url=text_base_url,
                model=environment.get("TEXT_MODEL", "gpt-5.6-terra").strip(),
                identity_namespace=_identity_namespace(environment, "TEXT", text_base_url),
            ),
            embedding=ProviderConfig(
                provider=environment.get("EMBEDDING_PROVIDER", "openai").strip().lower(),
                api_key=_optional(environment, "EMBEDDING_API_KEY") or openai_key,
                base_url=embedding_base_url,
                model=environment.get("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
                identity_namespace=_identity_namespace(
                    environment, "EMBEDDING", embedding_base_url
                ),
                dimensions=dimensions,
            ),
            narrative=ProviderConfig(
                provider=environment.get("NARRATIVE_PROVIDER", "openai").strip().lower(),
                api_key=_optional(environment, "NARRATIVE_API_KEY") or openai_key,
                base_url=narrative_base_url,
                model=environment.get(
                    "NARRATIVE_MODEL",
                    environment.get("TEXT_MODEL", "gpt-5.6-terra"),
                ).strip(),
                identity_namespace=_identity_namespace(
                    environment, "NARRATIVE", narrative_base_url
                ),
            ),
            bot=TelegramBotConfig(
                token=_optional(environment, "TELEGRAM_BOT_TOKEN"),
                owner_user_id=owner_user_id,
                channel=channel,
                api_base_url=_optional(environment, "TELEGRAM_BOT_API_BASE_URL"),
                api_file_url=_optional(environment, "TELEGRAM_BOT_API_FILE_URL"),
                max_download_bytes=_positive_int(
                    environment, "TELEGRAM_MAX_DOWNLOAD_BYTES", 20 * 1024 * 1024
                ),
            ),
            history=TelegramHistoryConfig(
                api_id=api_id,
                api_hash=_optional(environment, "TG_API_HASH"),
                phone=_optional(environment, "TG_PHONE"),
                session_path=Path(
                    environment.get(
                        "TG_SESSION_PATH", str(data_dir / "telegram" / "cancel_capture")
                    )
                ).expanduser(),
                channel=channel,
            ),
        )
