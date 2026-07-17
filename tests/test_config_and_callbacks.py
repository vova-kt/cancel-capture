import pytest

from cancel_capture.adapters.telegram_bot import parse_callback
from cancel_capture.config import AppConfig
from cancel_capture.errors import ConfigurationError


def test_provider_roles_inherit_openai_key_but_keep_independent_models(tmp_path) -> None:
    config = AppConfig.from_env(
        {
            "DATA_DIR": str(tmp_path / "data"),
            "OPENAI_API_KEY": "test-key",
            "VISION_MODEL": "vision-model",
            "TEXT_MODEL": "text-model",
            "EMBEDDING_MODEL": "embedding-model",
        },
        load_dotenv_file=False,
    )

    assert config.vision.api_key == "test-key"
    assert config.text.api_key == "test-key"
    assert config.embedding.api_key == "test-key"
    assert config.vision.model == "vision-model"
    assert config.text.model == "text-model"
    assert config.embedding.model == "embedding-model"
    assert config.bot.max_download_bytes == 20 * 1024 * 1024
    assert config.storage.max_upload_bytes == 100 * 1024 * 1024


def test_invalid_numeric_configuration_fails_early() -> None:
    with pytest.raises(ConfigurationError, match="TELEGRAM_OWNER_USER_ID"):
        AppConfig.from_env({"TELEGRAM_OWNER_USER_ID": "not-a-number"}, load_dotenv_file=False)


def test_empty_provider_or_model_configuration_fails_early() -> None:
    with pytest.raises(ConfigurationError, match="Provider name"):
        AppConfig.from_env({"VISION_PROVIDER": " "}, load_dotenv_file=False)
    with pytest.raises(ConfigurationError, match="Provider model"):
        AppConfig.from_env({"TEXT_MODEL": " "}, load_dotenv_file=False)


def test_custom_endpoint_changes_non_secret_provider_identity() -> None:
    first = AppConfig.from_env(
        {"EMBEDDING_BASE_URL": "https://one.example/v1"}, load_dotenv_file=False
    )
    second = AppConfig.from_env(
        {"EMBEDDING_BASE_URL": "https://two.example/v1"}, load_dotenv_file=False
    )

    assert first.embedding.identity_namespace.startswith("endpoint-")
    assert first.embedding.identity_namespace != second.embedding.identity_namespace


def test_bot_and_history_credentials_are_validated_separately(tmp_path) -> None:
    config = AppConfig.from_env({"DATA_DIR": str(tmp_path / "data")}, load_dotenv_file=False)
    with pytest.raises(ConfigurationError, match="TELEGRAM_BOT_TOKEN"):
        config.bot.validate()
    with pytest.raises(ConfigurationError, match="TG_API_ID"):
        config.history.validate()


def test_callback_parser_accepts_only_compact_opaque_ids() -> None:
    item_id = "a" * 32
    token = "b" * 16
    assert parse_callback(f"publish:{item_id}:{token}") == ("publish", item_id, token)
    assert parse_callback(f"retry:{item_id}:{token}") == ("retry", item_id, token)
    assert parse_callback(f"reject:{item_id}:{token}") == ("reject", item_id, token)
    assert parse_callback(f"publish:{item_id}") is None
    assert parse_callback("publish:1 OR 1=1") is None
    assert parse_callback(f"delete:{item_id}:{token}") is None
    assert parse_callback(f"publish:{'a' * 33}:{token}") is None
