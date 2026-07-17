from __future__ import annotations

from typing import cast

import streamlit as st

from cancel_capture.application import (
    NarrativeExperimentRequest,
    NarrativeSelection,
    SimilarityMode,
    default_system_prompt,
)
from cancel_capture.container import NarrativeServices, Services
from cancel_capture.errors import CancelCaptureError
from cancel_capture.models import ReviewStatus, SignEmbeddingDocument
from cancel_capture.narrative_models import NarrativeLanguage, StoredNarrativeArtifact
from cancel_capture.prompts import NarrativeStrategy
from cancel_capture.streamlitapp._shared import (
    DEFAULT_STATUSES,
    LANGUAGE_LABELS,
    SIMILARITY_LABELS,
    STRATEGY_LABELS,
    StreamlitProgress,
    narrative_services,
    new_seed,
    run,
)


def render(services: Services, narrative: NarrativeServices) -> None:
    session = st.session_state
    session.setdefault("narrative.system_prompt", default_system_prompt())
    session.setdefault("narrative.similarity_mode", SimilarityMode.HYBRID.value)
    session.setdefault("narrative.similarity_threshold", 0.55)
    session.setdefault("narrative.semantic_weight", 0.65)
    session.setdefault("narrative.anchor_weight", 2.5)

    statuses = DEFAULT_STATUSES
    anchors = _anchor_options_cached(statuses)
    if not anchors:
        st.info("Ingest at least one sign before generating a narrative.")
        return

    controls = st.container()
    with controls:
        left, right = st.columns([2, 1])
        with left:
            options = list(anchors)
            option_ids = [document.item_id for document in options]
            current = str(session.get("narrative.anchor_id", option_ids[0]))
            if current not in option_ids:
                current = option_ids[0]
            selected_id = st.selectbox(
                "Anchor sign",
                options=option_ids,
                index=option_ids.index(current),
                format_func=lambda value: _anchor_label(
                    next(document for document in options if document.item_id == value)
                ),
            )
            session["narrative.anchor_id"] = selected_id
        with right:
            if st.button("Random anchor", width="stretch"):
                seed = new_seed()
                choice = narrative.selection.random_anchor(statuses, seed=seed)
                if choice is not None:
                    session["narrative.anchor_id"] = choice.item_id
                    st.rerun()

        settings = st.columns(4)
        with settings[0]:
            companion_count = st.slider(
                "Companion signs (n)",
                min_value=1,
                max_value=max(1, min(12, len(anchors) - 1)),
                value=int(cast(int, session.get("narrative.companion_count", 4))),
            )
            session["narrative.companion_count"] = companion_count
        with settings[1]:
            reading_minutes = st.slider(
                "Reading minutes",
                min_value=1,
                max_value=20,
                value=int(cast(int, session.get("narrative.reading_minutes", 2))),
            )
            session["narrative.reading_minutes"] = reading_minutes
        with settings[2]:
            language_value = st.selectbox(
                "Language",
                options=[language.value for language in NarrativeLanguage],
                format_func=lambda value: LANGUAGE_LABELS[NarrativeLanguage(value)],
                index=[language.value for language in NarrativeLanguage].index(
                    str(session.get("narrative.language", NarrativeLanguage.ENGLISH.value))
                ),
            )
            session["narrative.language"] = language_value
        with settings[3]:
            strategy_value = st.selectbox(
                "Strategy",
                options=[strategy.value for strategy in NarrativeStrategy],
                format_func=lambda value: STRATEGY_LABELS[NarrativeStrategy(value)],
                index=[strategy.value for strategy in NarrativeStrategy].index(
                    str(session.get("narrative.strategy", NarrativeStrategy.FAMILY_CHRONICLE.value))
                ),
            )
            session["narrative.strategy"] = strategy_value

        actions = st.columns([1, 1, 1, 1])
        with actions[0]:
            if st.button("Advanced options"):
                _advanced_options_dialog()
        with actions[1]:
            if st.button("Edit system prompt"):
                _system_prompt_dialog()
        with actions[2]:
            if st.button("Resample companions"):
                session["narrative.selection_seed"] = new_seed()
        with actions[3]:
            generate = st.button("Generate story", type="primary")

    session.setdefault("narrative.selection_seed", new_seed())
    selection = narrative.selection.select(
        str(session["narrative.anchor_id"]),
        count=int(cast(int, session["narrative.companion_count"])),
        maximum_similarity=float(cast(float, session["narrative.similarity_threshold"])),
        statuses=statuses,
        mode=SimilarityMode(str(session["narrative.similarity_mode"])),
        semantic_weight=float(cast(float, session["narrative.semantic_weight"])),
        seed=int(cast(int, session["narrative.selection_seed"])),
    )
    _render_selection_preview(services, selection)

    if generate:
        if not selection.companions:
            st.error(
                "No eligible companion signs matched the similarity filter. "
                "Ingest more signs or loosen the constraints before generating."
            )
            return
        request = NarrativeExperimentRequest(
            selection=selection,
            strategy=NarrativeStrategy(str(session["narrative.strategy"])),
            language=NarrativeLanguage(str(session["narrative.language"])),
            reading_minutes=int(cast(int, session["narrative.reading_minutes"])),
            system_prompt=str(session["narrative.system_prompt"]),
            similarity_mode=SimilarityMode(str(session["narrative.similarity_mode"])),
            similarity_threshold=float(cast(float, session["narrative.similarity_threshold"])),
            semantic_weight=float(cast(float, session["narrative.semantic_weight"])),
            anchor_weight=float(cast(float, session["narrative.anchor_weight"])),
            random_seed=int(cast(int, session["narrative.selection_seed"])),
            news_query=selection.anchor.document.text.en,
        )
        try:
            with st.status("Starting…", expanded=True) as status:
                reporter = StreamlitProgress(status)
                result = run(narrative.experiments.generate(request, progress=reporter))
        except (CancelCaptureError, ValueError) as error:
            st.error(f"Narrative generation failed: {error}")
        else:
            session["narrative.last_result"] = result.stored.relative_path
            st.success(f"Saved to `{result.stored.relative_path}`.")

    _render_last_result(narrative)
    st.divider()
    _render_saved_narratives(narrative)


@st.dialog("Advanced narrative options")
def _advanced_options_dialog() -> None:
    st.markdown("Change how companions are picked and how the anchor is emphasized.")
    session = st.session_state
    session.setdefault("narrative.similarity_mode", SimilarityMode.HYBRID.value)
    session.setdefault("narrative.similarity_threshold", 0.55)
    session.setdefault("narrative.semantic_weight", 0.65)
    session.setdefault("narrative.anchor_weight", 2.5)
    session["narrative.similarity_mode"] = st.selectbox(
        "Similarity mode",
        options=[mode.value for mode in SimilarityMode],
        index=[mode.value for mode in SimilarityMode].index(
            str(session["narrative.similarity_mode"])
        ),
        format_func=lambda value: SIMILARITY_LABELS[SimilarityMode(value)],
        help="Hybrid mixes semantic and visual similarity; falls back if a vector is missing.",
    )
    session["narrative.similarity_threshold"] = st.slider(
        "Maximum similarity to anchor",
        min_value=-1.0,
        max_value=1.0,
        step=0.05,
        value=float(cast(float, session["narrative.similarity_threshold"])),
        help="Companions must fall at or below this cosine similarity.",
    )
    session["narrative.semantic_weight"] = st.slider(
        "Semantic weight (hybrid)",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=float(cast(float, session["narrative.semantic_weight"])),
    )
    session["narrative.anchor_weight"] = st.slider(
        "Anchor weight vs. companions",
        min_value=1.0,
        max_value=6.0,
        step=0.25,
        value=float(cast(float, session["narrative.anchor_weight"])),
        help="Higher values push the model to lean harder on the anchor sign.",
    )
    if st.button("Done"):
        st.rerun()


@st.dialog("Narrative system prompt")
def _system_prompt_dialog() -> None:
    session = st.session_state
    session.setdefault("narrative.system_prompt", default_system_prompt())
    session["narrative.system_prompt"] = st.text_area(
        "System prompt",
        value=str(session["narrative.system_prompt"]),
        height=360,
    )
    columns = st.columns(2)
    with columns[0]:
        if st.button("Reset to default"):
            session["narrative.system_prompt"] = default_system_prompt()
            st.rerun()
    with columns[1]:
        if st.button("Done"):
            st.rerun()


def _anchor_label(document: SignEmbeddingDocument) -> str:
    excerpt = document.text.en.strip().splitlines()[0]
    prefix = document.item_id[:8]
    return f"{prefix} · {excerpt[:80]}"


@st.cache_data(show_spinner=False, ttl=30)
def _anchor_options_cached(
    statuses: frozenset[ReviewStatus],
) -> tuple[SignEmbeddingDocument, ...]:
    return narrative_services().selection.list_anchors(statuses)


def _render_selection_preview(services: Services, selection: NarrativeSelection) -> None:
    st.markdown("#### Selection preview")
    st.caption(
        f"{selection.eligible_count} eligible companions, "
        f"{len(selection.companions)}/{selection.requested_count} sampled"
    )
    companion_entries: list[tuple[str, str]] = [
        (
            companion.document.asset_relative_path,
            f"cos={companion.similarity_to_anchor:+.2f}",
        )
        for companion in selection.companions
    ]
    signs: list[tuple[str, str]] = [
        (selection.anchor.document.asset_relative_path, "Anchor (weighted)"),
        *companion_entries,
    ]
    columns = st.columns(max(1, len(signs)))
    for column, (relative_path, caption) in zip(columns, signs, strict=True):
        with column:
            st.image(str(services.assets.resolve(relative_path)), caption=caption)


def _render_last_result(narrative: NarrativeServices) -> None:
    session = st.session_state
    relative_path = session.get("narrative.last_result")
    if not isinstance(relative_path, str):
        return
    try:
        stored = narrative.experiments.read_saved(relative_path)
    except (ValueError, FileNotFoundError) as error:
        st.warning(f"Could not read the saved narrative: {error}")
        return
    _render_artifact(stored)


def _render_saved_narratives(narrative: NarrativeServices) -> None:
    st.markdown("### Saved narratives")
    artifacts = narrative.experiments.list_saved()
    if not artifacts:
        st.caption("Nothing saved yet.")
        return
    labels = [
        f"{artifact.artifact.metadata.created_at} · {artifact.artifact.title}"
        for artifact in artifacts
    ]

    def _artifact_label(position: int) -> str:
        return labels[position]

    index = st.selectbox(
        "Open a saved story",
        options=list(range(len(artifacts))),
        format_func=_artifact_label,
    )
    _render_artifact(artifacts[int(index)])


def _render_artifact(stored: StoredNarrativeArtifact) -> None:
    artifact = stored.artifact
    metadata = artifact.metadata
    st.markdown(f"## {artifact.title}")
    st.caption(
        f"{metadata.language} · {metadata.reading_minutes} min · strategy: {metadata.strategy}"
    )
    st.markdown(f"*{artifact.description}*")
    st.markdown(artifact.body_markdown)
    with st.expander("Metadata and prompts"):
        st.caption(f"Attempt ID: `{metadata.attempt_id}`")
        st.caption(f"Saved at: `{stored.relative_path}`")
        st.caption(f"Anchor: `{metadata.anchor_sign_id}`")
        if metadata.source_sign_ids:
            st.caption(f"Companions: {', '.join(metadata.source_sign_ids)}")
        if metadata.web_citations:
            st.markdown("**Citations**")
            for citation in metadata.web_citations:
                st.markdown(f"- [{citation.title}]({citation.url})")
        st.markdown("**System prompt**")
        st.code(metadata.system_prompt)
        st.markdown("**User prompt**")
        st.code(metadata.user_prompt)
