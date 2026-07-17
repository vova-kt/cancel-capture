from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_starts_and_reports_missing_provider_configuration(monkeypatch, tmp_path) -> None:
    for name in (
        "OPENAI_API_KEY",
        "VISION_API_KEY",
        "TEXT_API_KEY",
        "EMBEDDING_API_KEY",
        "NARRATIVE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("cancel_capture.config.load_dotenv", lambda *_args, **_kwargs: False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "data" / "catalog.sqlite3"))

    app_path = Path(__file__).parents[1] / "src" / "cancel_capture" / "streamlitapp" / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    assert not app.exception
    assert app.error
    assert "Configuration error" in app.error[0].value
