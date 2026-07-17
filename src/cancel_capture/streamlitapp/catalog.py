from __future__ import annotations

import html
import json
from typing import cast

import streamlit as st

from cancel_capture.application import (
    NarrativeExperimentRequest,
    SimilarityMode,
    default_system_prompt,
)
from cancel_capture.container import Services
from cancel_capture.errors import CancelCaptureError
from cancel_capture.models import ReviewCandidate
from cancel_capture.narrative_models import NarrativeLanguage, StoredNarrativeArtifact
from cancel_capture.prompts import NarrativeStrategy
from cancel_capture.streamlitapp._shared import (
    DEFAULT_STATUSES,
    LANGUAGE_LABELS,
    STRATEGY_LABELS,
    StreamlitProgress,
    core_services,
    image_to_data_uri,
    narrative_services,
    new_seed,
    run,
)

_GRID_COLUMNS = 4
_LATEST_LIMIT = 100
_THUMB_HEIGHT_PX = 160
_THUMB_MAX_BYTES = 2_000_000


def render(services: Services) -> None:
    candidates = services.catalog.list_candidates()
    recent = list(reversed(candidates[-_LATEST_LIMIT:]))
    st.caption(f"{len(candidates)} sign candidates (showing latest {len(recent)})")
    if not recent:
        st.info("Ingest a sign to populate the catalog.")
        return
    for row_start in range(0, len(recent), _GRID_COLUMNS):
        row = recent[row_start : row_start + _GRID_COLUMNS]
        columns = st.columns(_GRID_COLUMNS, gap="small")
        for column, candidate in zip(columns, row, strict=False):
            with column:
                _grid_cell(services, candidate)


def _grid_cell(services: Services, candidate: ReviewCandidate) -> None:
    with st.container(border=True):
        _thumbnail(services, candidate)
        description = _first_line(candidate.sign_description.en) or "(no description)"
        st.markdown(
            f'<div style="font-size:0.85rem;line-height:1.25;height:2.5em;'
            f"overflow:hidden;margin:0.4rem 0 0.2rem;"
            f'">{html.escape(description)}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"`{candidate.item_id[:8]}` · {candidate.status.value}")
        if st.button("Open", key=f"catalog-open-{candidate.item_id}", width="stretch"):
            _candidate_dialog(candidate.item_id)


def _thumbnail(services: Services, candidate: ReviewCandidate) -> None:
    asset_path = str(services.assets.resolve(candidate.crop_relative_path))
    thumb_uri = image_to_data_uri(asset_path, max_bytes=_THUMB_MAX_BYTES)
    if thumb_uri is None:
        # Files bigger than the inline budget fall back to Streamlit's static server;
        # they lose the fixed-height layout but stay visible.
        st.image(asset_path, width=200)
        return
    st.markdown(
        f'<div style="height:{_THUMB_HEIGHT_PX}px;overflow:hidden;border-radius:6px;'
        f"background:#f6f6f6;display:flex;align-items:center;justify-content:center;"
        f'">'
        f'<img src="{thumb_uri}" alt="sign" '
        f'style="max-height:100%;max-width:100%;object-fit:contain;"></div>',
        unsafe_allow_html=True,
    )


def _first_line(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.splitlines()[0][:80]


@st.dialog("Sign details", width="large")
def _candidate_dialog(item_id: str) -> None:
    services = core_services()
    candidate = services.catalog.get_candidate(item_id)
    st.image(str(services.assets.resolve(candidate.crop_relative_path)))
    st.markdown(f"### {candidate.sign_description.en}")
    st.caption(
        f"Status: {candidate.status.value} · sign #{candidate.ordinal + 1} · `{candidate.item_id}`"
    )
    if candidate.topics_en:
        st.write(f"Topics: {', '.join(candidate.topics_en)}")
    with st.expander("Metadata"):
        st.json(cast(object, json.loads(candidate.metadata.raw_json)))
    st.divider()
    _narrative_form(item_id)


def _narrative_form(anchor_id: str) -> None:
    st.markdown("### Generate narrative from this sign")
    try:
        narrative = narrative_services()
    except (CancelCaptureError, ValueError) as error:
        st.warning(f"Narrative provider is not configured: {error}")
        return

    columns = st.columns(4)
    with columns[0]:
        companion_count = st.slider(
            "Companions",
            min_value=1,
            max_value=8,
            value=4,
            key="catalog.narrative.companions",
        )
    with columns[1]:
        reading_minutes = st.slider(
            "Minutes",
            min_value=1,
            max_value=20,
            value=2,
            key="catalog.narrative.minutes",
        )
    with columns[2]:
        language_value = st.selectbox(
            "Language",
            options=[language.value for language in NarrativeLanguage],
            format_func=lambda value: LANGUAGE_LABELS[NarrativeLanguage(value)],
            key="catalog.narrative.language",
        )
    with columns[3]:
        strategy_value = st.selectbox(
            "Strategy",
            options=[strategy.value for strategy in NarrativeStrategy],
            format_func=lambda value: STRATEGY_LABELS[NarrativeStrategy(value)],
            key="catalog.narrative.strategy",
        )

    result_key = f"catalog.narrative.result.{anchor_id}"
    if st.button("Generate story", type="primary", key=f"catalog-gen-{anchor_id}"):
        seed = new_seed()
        try:
            selection = narrative.selection.select(
                anchor_id,
                count=int(companion_count),
                maximum_similarity=0.55,
                statuses=DEFAULT_STATUSES,
                mode=SimilarityMode.HYBRID,
                semantic_weight=0.65,
                seed=seed,
            )
            if not selection.companions:
                st.error(
                    "No eligible companion signs matched the similarity filter. "
                    "Ingest more signs or loosen the constraints before generating."
                )
                return
            request = NarrativeExperimentRequest(
                selection=selection,
                strategy=NarrativeStrategy(str(strategy_value)),
                language=NarrativeLanguage(str(language_value)),
                reading_minutes=int(reading_minutes),
                system_prompt=default_system_prompt(),
                similarity_mode=SimilarityMode.HYBRID,
                similarity_threshold=0.55,
                semantic_weight=0.65,
                anchor_weight=2.5,
                random_seed=seed,
                news_query=selection.anchor.document.text.en,
            )
            with st.status("Generating…", expanded=True) as status:
                reporter = StreamlitProgress(status)
                result = run(narrative.experiments.generate(request, progress=reporter))
        except (CancelCaptureError, ValueError) as error:
            st.error(f"Narrative generation failed: {error}")
        else:
            st.session_state[result_key] = result.stored

    stored = st.session_state.get(result_key)
    if isinstance(stored, StoredNarrativeArtifact):
        _render_artifact(stored)


def _render_artifact(stored: StoredNarrativeArtifact) -> None:
    artifact = stored.artifact
    metadata = artifact.metadata
    st.markdown(f"## {artifact.title}")
    st.caption(
        f"{metadata.language} · {metadata.reading_minutes} min · strategy: {metadata.strategy}"
    )
    st.markdown(f"*{artifact.description}*")
    st.markdown(artifact.body_markdown)
    st.caption(f"Saved to `{stored.relative_path}`")
