from __future__ import annotations

import stat
from pathlib import Path

import pytest

from cancel_capture.adapters.markdown_narratives import (
    MarkdownNarrativeStore,
    NarrativeArtifactFormatError,
)
from cancel_capture.narrative_models import (
    NarrativeArtifact,
    NarrativeArtifactMetadata,
    WebCitation,
)


def _make_artifact(**overrides: object) -> NarrativeArtifact:
    metadata = NarrativeArtifactMetadata.create(
        anchor_sign_id=str(overrides.pop("anchor", "anchor")),
        source_sign_ids=("companion-1", "companion-2"),
        strategy="family_chronicle",
        language="English",
        reading_minutes=2,
        similarity_mode="hybrid",
        similarity_threshold=0.55,
        semantic_weight=0.65,
        random_seed=17,
        provider="openai",
        model="gpt-5.6-terra",
        provider_namespace="default",
        system_prompt="A system prompt used for testing.",
        user_prompt="A user prompt used for testing.",
        web_citations=(WebCitation(title="Example", url="https://example.com"),),
    )
    return NarrativeArtifact(
        title=str(overrides.pop("title", "A quiet extinction")),
        description=str(overrides.pop("description", "A short future about bans.")),
        body_markdown=str(
            overrides.pop("body", "# One\n\nFirst paragraph.\n\n## Two\n\nSecond paragraph.")
        ),
        metadata=metadata,
    )


def test_saved_artifact_round_trips_through_read_and_list(tmp_path: Path) -> None:
    store = MarkdownNarrativeStore(tmp_path)
    artifact = _make_artifact()

    stored = store.save(artifact)

    assert stored.artifact == artifact
    on_disk = store.read(stored.relative_path)
    assert on_disk == stored

    listed = store.list_artifacts()
    assert listed == (stored,)


def test_saved_file_is_private_and_lives_under_narratives(tmp_path: Path) -> None:
    store = MarkdownNarrativeStore(tmp_path)
    stored = store.save(_make_artifact())

    absolute = tmp_path / stored.relative_path
    assert stored.relative_path.startswith("narratives/")
    assert absolute.exists()
    mode = stat.S_IMODE(absolute.stat().st_mode)
    assert mode == 0o600
    directory_mode = stat.S_IMODE((tmp_path / "narratives").stat().st_mode)
    assert directory_mode == 0o700


def test_multiple_saves_sort_newest_first(tmp_path: Path) -> None:
    store = MarkdownNarrativeStore(tmp_path)
    first = store.save(_make_artifact(title="First"))
    second = store.save(_make_artifact(title="Second"))

    listed = store.list_artifacts()

    ordered = [entry.artifact.metadata.created_at for entry in listed]
    assert ordered == sorted(ordered, reverse=True)
    assert {entry.artifact.title for entry in listed} == {"First", "Second"}
    assert {first.artifact.metadata.attempt_id, second.artifact.metadata.attempt_id} == {
        entry.artifact.metadata.attempt_id for entry in listed
    }


def test_corrupted_front_matter_is_rejected_safely(tmp_path: Path) -> None:
    store = MarkdownNarrativeStore(tmp_path)
    stored = store.save(_make_artifact())

    target = tmp_path / stored.relative_path
    target.write_text("not front matter", encoding="utf-8")
    with pytest.raises(NarrativeArtifactFormatError):
        store.read(stored.relative_path)


def test_traversal_and_hidden_paths_are_rejected(tmp_path: Path) -> None:
    store = MarkdownNarrativeStore(tmp_path)

    with pytest.raises(ValueError):
        store.read("narratives/../secret.md")
    with pytest.raises(ValueError):
        store.read("outside/whatever.md")
    with pytest.raises(NarrativeArtifactFormatError):
        store.read("narratives/does-not-exist.md")
