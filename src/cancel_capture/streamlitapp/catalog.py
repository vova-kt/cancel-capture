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
from cancel_capture.models import ReviewCandidate, ReviewStatus
from cancel_capture.narrative_models import NarrativeLanguage, StoredNarrativeArtifact
from cancel_capture.prompts import NarrativeStrategy
from cancel_capture.streamlitapp._shared import (
    DEFAULT_STATUSES,
    LANGUAGE_LABELS,
    SIMILARITY_LABELS,
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

    mode, semantic_weight, threshold = _similarity_controls(anchor_id)

    result_key = f"catalog.narrative.result.{anchor_id}"
    if st.button("Generate story", type="primary", key=f"catalog-gen-{anchor_id}"):
        seed = new_seed()
        try:
            selection = narrative.selection.select(
                anchor_id,
                count=int(companion_count),
                maximum_similarity=threshold,
                statuses=DEFAULT_STATUSES,
                mode=mode,
                semantic_weight=semantic_weight,
                seed=seed,
            )
            if not selection.companions:
                st.error(
                    "No eligible companion signs matched the similarity filter. "
                    "Loosen the threshold or switch similarity mode."
                )
                return
            request = NarrativeExperimentRequest(
                selection=selection,
                strategy=NarrativeStrategy(str(strategy_value)),
                language=NarrativeLanguage(str(language_value)),
                reading_minutes=int(reading_minutes),
                system_prompt=default_system_prompt(),
                similarity_mode=mode,
                similarity_threshold=threshold,
                semantic_weight=semantic_weight,
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


def _similarity_controls(anchor_id: str) -> tuple[SimilarityMode, float, float]:
    """Render the similarity mode / weight / threshold row with a live match-count preview.

    The similarity pool is cached per anchor+mode+weight so dragging the threshold slider
    only re-scans an in-memory tuple — no catalog scan or model call fires per slider tick.
    """
    row = st.columns([1.2, 1.2, 2])
    with row[0]:
        mode_value = st.selectbox(
            "Similarity mode",
            options=[m.value for m in SimilarityMode],
            format_func=lambda value: SIMILARITY_LABELS[SimilarityMode(value)],
            index=[m.value for m in SimilarityMode].index(SimilarityMode.SEMANTIC.value),
            key="catalog.narrative.mode",
        )
    mode = SimilarityMode(str(mode_value))

    with row[1]:
        if mode is SimilarityMode.HYBRID:
            semantic_weight = float(
                st.slider(
                    "Semantic weight",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    value=0.65,
                    key="catalog.narrative.semantic_weight",
                )
            )
        else:
            st.caption("Semantic weight applies only to Hybrid mode.")
            semantic_weight = 0.65

    similarities = _eligible_similarities(anchor_id, mode, semantic_weight, DEFAULT_STATUSES)
    initial = _suggest_threshold(similarities)

    with row[2]:
        threshold = float(
            st.slider(
                "Max similarity to anchor",
                min_value=-1.0,
                max_value=1.0,
                step=0.01,
                value=initial,
                key="catalog.narrative.threshold",
                help=(
                    "Companions must fall at or below this cosine similarity to the anchor "
                    "(lower ⇒ more diverse)."
                ),
            )
        )

    _render_pool_preview(similarities, threshold, mode)
    return mode, semantic_weight, threshold


@st.cache_data(show_spinner=False, ttl=60)
def _eligible_similarities(
    anchor_id: str,
    mode: SimilarityMode,
    semantic_weight: float,
    statuses: frozenset[ReviewStatus],
) -> tuple[float, ...]:
    return narrative_services().selection.list_eligible_similarities(
        anchor_id,
        statuses=statuses,
        mode=mode,
        semantic_weight=semantic_weight,
    )


def _suggest_threshold(similarities: tuple[float, ...]) -> float:
    if not similarities:
        return 0.9
    # 25th percentile of the pool — keeps companions on the more-dissimilar side without
    # pruning so aggressively that the sample can't fill.
    return round(similarities[max(0, len(similarities) // 4)], 2)


def _render_pool_preview(
    similarities: tuple[float, ...],
    threshold: float,
    mode: SimilarityMode,
) -> None:
    if not similarities:
        if mode is SimilarityMode.HYBRID:
            st.warning(
                "Hybrid mode found no candidates — visual embeddings may be missing. "
                "Run `scripts/backfill_visual_embeddings.py` or switch mode."
            )
        elif mode is SimilarityMode.VISUAL:
            st.warning(
                "No visual embeddings on record. "
                "Run `scripts/backfill_visual_embeddings.py` to populate them."
            )
        else:
            st.warning("No structurally eligible companions for this anchor.")
        return
    matches = sum(1 for value in similarities if value <= threshold)
    median = similarities[len(similarities) // 2]
    st.caption(
        f"**{matches}** eligible companion(s) at ≤ {threshold:.2f} · "
        f"pool n={len(similarities)} · min={similarities[0]:.2f} · "
        f"median={median:.2f} · max={similarities[-1]:.2f}"
    )


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
