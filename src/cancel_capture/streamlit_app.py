from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import json
import secrets
from collections.abc import Awaitable, Sequence
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

import altair as alt
import pandas as pd  # pyright: ignore [reportMissingTypeStubs]
import streamlit as st

from cancel_capture.application import (
    NarrativeExperimentRequest,
    NarrativeSelection,
    SimilarityMode,
    default_system_prompt,
)
from cancel_capture.application.clustering import (
    DendrogramGeometry,
    HierarchicalClustering,
    average_linkage_cosine,
)
from cancel_capture.config import AppConfig
from cancel_capture.container import (
    NarrativeServices,
    Services,
    build_narrative_services,
    build_services,
)
from cancel_capture.errors import CancelCaptureError
from cancel_capture.models import (
    IngestionResult,
    IngestRequest,
    ItemKind,
    ReviewStatus,
    SearchDocument,
    SignEmbeddingDocument,
    SourceKind,
)
from cancel_capture.narrative_models import (
    NarrativeLanguage,
    StoredNarrativeArtifact,
)
from cancel_capture.prompts import NarrativeStrategy

DEFAULT_STATUSES = frozenset({ReviewStatus.PUBLISHED, ReviewStatus.PENDING})
_LANGUAGE_LABELS: dict[NarrativeLanguage, str] = {
    NarrativeLanguage.ENGLISH: "English",
    NarrativeLanguage.RUSSIAN: "Русский",
}
_STRATEGY_LABELS: dict[NarrativeStrategy, str] = {
    NarrativeStrategy.FAMILY_CHRONICLE: "Family chronicle",
    NarrativeStrategy.CIVIC_RIPPLE: "Civic ripple",
    NarrativeStrategy.NEWS_MONTAGE: "News montage",
    NarrativeStrategy.EVERYDAY_ADAPTATION: "Everyday adaptation",
}
_SIMILARITY_LABELS: dict[SimilarityMode, str] = {
    SimilarityMode.HYBRID: "Hybrid (semantic + visual)",
    SimilarityMode.SEMANTIC: "Semantic only",
    SimilarityMode.VISUAL: "Visual only",
}


def _core_services() -> Services:
    return build_services(AppConfig.from_env())


def _narrative_services(services: Services) -> NarrativeServices:
    return build_narrative_services(services)


def _new_seed() -> int:
    return secrets.randbits(31)


def _run[T](coroutine: Awaitable[T]) -> T:
    return asyncio.run(_await(coroutine))


async def _await[T](coroutine: Awaitable[T]) -> T:
    return await coroutine


class StreamlitProgress:
    """Bridge our ProgressReporter protocol onto a Streamlit ``st.status`` container.

    ``st.status`` streams children to the browser as the script executes, so ``note`` and
    ``stage`` calls made during an ``asyncio.run`` inner loop reach the user in near real time.
    """

    def __init__(self, status: object) -> None:
        self._status = cast("_StatusApi", status)

    def stage(self, key: str, label: str) -> None:
        self._status.update(label=label, state="running")

    def note(self, text: str) -> None:
        self._status.write(f"· {text}")

    def complete(self, label: str, *, ok: bool = True) -> None:
        self._status.update(label=label, state="complete" if ok else "error")


class _StatusApi(Protocol):
    def update(self, *, label: str, state: str) -> None: ...

    def write(self, body: object) -> None: ...


def _sign_documents(services: Services) -> tuple[SearchDocument, ...]:
    signs = services.catalog.list_sign_embedding_documents()
    return tuple(
        SearchDocument(
            item_id=document.item_id,
            kind=ItemKind.SIGN,
            text=document.text,
            asset_relative_path=document.asset_relative_path,
            embedding=document.semantic_embedding,
            status=document.status,
        )
        for document in signs
    )


def _image_to_data_uri(path: Path, *, max_bytes: int = 512_000) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > max_bytes:
        return None
    suffix = path.suffix.lstrip(".").lower() or "jpeg"
    media_type = {"jpg": "jpeg", "jpe": "jpeg"}.get(suffix, suffix)
    return f"data:image/{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def _cluster_thumb_html(services: Services, document: SearchDocument) -> str:
    data_uri = _image_to_data_uri(services.assets.resolve(document.asset_relative_path))
    if data_uri is None:
        return ""
    tooltip = html.escape(
        f"{document.text.en}\n\n{document.text.ru}",
        quote=True,
    )
    return (
        f'<img src="{data_uri}" title="{tooltip}" alt="sign" '
        'style="height:96px;margin:4px;border-radius:6px;'
        'box-shadow:0 1px 4px rgba(0,0,0,0.15);object-fit:cover">'
    )


def _dendrogram_chart(geometry: DendrogramGeometry, cut_distance: float | None) -> alt.LayerChart:
    if not geometry.segments:
        empty = alt.Chart(pd.DataFrame({"x": [0.0], "y": [0.0]})).mark_point()
        return cast(alt.LayerChart, alt.layer(empty).properties(height=280))
    frame = pd.DataFrame(
        [
            {
                "x": segment.x_start,
                "y": segment.y_start,
                "x2": segment.x_end,
                "y2": segment.y_end,
                "merge": segment.merge_node_id,
            }
            for segment in geometry.segments
        ]
    )
    lines = (
        alt.Chart(frame)
        .mark_rule(color="#4c78a8")
        .encode(
            x=alt.X("x:Q", axis=alt.Axis(title="Leaf index")),
            y=alt.Y("y:Q", axis=alt.Axis(title="Cosine distance")),
            x2="x2:Q",
            y2="y2:Q",
        )
    )
    if cut_distance is None:
        return cast(alt.LayerChart, alt.layer(lines).properties(height=280))
    rule = (
        alt.Chart(pd.DataFrame({"y": [cut_distance]}))
        .mark_rule(color="#e45756", strokeDash=[6, 4])
        .encode(y="y:Q")
    )
    return cast(alt.LayerChart, alt.layer(lines, rule).properties(height=280))


def _analyze_tab(services: Services) -> None:
    st.write(
        "Upload the same unprocessed image file that you would send to Telegram. The original is "
        "stored in the configured data volume."
    )
    uploaded = st.file_uploader(
        "Original image / Оригинал",
        type=["jpg", "jpeg", "png", "webp", "heic", "heif", "tif", "tiff"],
    )
    reanalyze = st.checkbox(
        "Reanalyze as a new run (uses provider API calls)",
        help=(
            "Enable this when comparing prompt, model, or crop changes on the same reference file."
        ),
    )
    if uploaded is not None and st.button("Analyze and crop / Анализировать", type="primary"):
        data = uploaded.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        source_key = f"streamlit:{digest}:{uuid4().hex}" if reanalyze else f"streamlit:{digest}"
        with st.status("Analyzing…", expanded=True) as status:
            result = _run(
                services.ingestion.ingest(
                    IngestRequest(
                        data=data,
                        filename=uploaded.name,
                        declared_media_type=uploaded.type,
                        source_kind=SourceKind.STREAMLIT,
                        source_key=source_key,
                    )
                )
            )
            status.update(label="Analysis complete", state="complete")
        st.session_state["last_result"] = result
    result = st.session_state.get("last_result")
    if isinstance(result, IngestionResult):
        _show_result(services, result)


def _show_result(services: Services, result: IngestionResult) -> None:
    st.subheader("Full photo / Полное фото")
    st.image(str(services.assets.resolve(result.original_relative_path)))
    st.write(f"**English:** {result.description.en}")
    st.write(f"**Русский:** {result.description.ru}")
    metadata = cast(object, json.loads(result.metadata.raw_json))
    with st.expander("All extracted metadata / Все метаданные"):
        st.json(metadata)

    st.subheader(f"Detected signs / Обнаруженные знаки: {len(result.signs)}")
    if not result.signs:
        st.info("No qualifying prohibition sign was detected. The full photo is still archived.")
    for sign in result.signs:
        candidate = services.catalog.get_candidate(sign.item_id)
        left, right = st.columns([1, 2])
        with left:
            st.image(str(services.assets.resolve(candidate.crop_relative_path)))
            st.caption(f"Status: {candidate.status.value} · ID: {candidate.item_id}")
        with right:
            st.write(f"**English:** {candidate.sign_description.en}")
            st.write(f"**Русский:** {candidate.sign_description.ru}")
            st.write(f"Topics: {', '.join(candidate.topics_en)}")
            st.write(f"Темы: {', '.join(candidate.topics_ru)}")
            if candidate.status in (ReviewStatus.PENDING, ReviewStatus.FAILED) and st.button(
                "Reject locally / Отклонить локально",
                key=f"reject-{candidate.item_id}",
            ):
                services.review.reject(candidate.item_id, candidate.review_token, actor_user_id=0)
                st.rerun()
            st.caption("Channel publishing remains behind the Telegram confirmation button.")


def _search_tab(services: Services) -> None:
    query = st.text_input("Topic in English or Russian / Тема на английском или русском")
    scope = st.selectbox("Scope", ["all", ItemKind.SIGN.value, ItemKind.PHOTO.value])
    if query.strip() and st.button("Search / Искать"):
        kind = None if scope == "all" else ItemKind(scope)
        hits = _run(services.search.search(query, kind=kind, limit=30))
        st.caption(f"{len(hits)} results")
        for hit in hits:
            left, right = st.columns([1, 3])
            with left:
                st.image(str(services.assets.resolve(hit.asset_relative_path)))
            with right:
                st.write(f"**{hit.score:.3f} · {hit.kind.value} · {hit.status.value}**")
                st.write(hit.description.en)
                st.write(hit.description.ru)


def _catalog_tab(services: Services) -> None:
    candidates = services.catalog.list_candidates()
    st.caption(f"{len(candidates)} sign candidates")
    for candidate in reversed(candidates[-100:]):
        with st.expander(
            f"{candidate.status.value} · sign {candidate.ordinal + 1} · {candidate.item_id[:8]}"
        ):
            st.image(str(services.assets.resolve(candidate.crop_relative_path)), width=300)
            st.write(candidate.sign_description.en)
            st.write(candidate.sign_description.ru)
            st.json(cast(object, json.loads(candidate.metadata.raw_json)))


def _clusters_tab(services: Services, narrative: NarrativeServices) -> None:
    documents = _sign_documents(services)
    st.caption(f"{len(documents)} sign documents with semantic embeddings")
    if len(documents) < 2:
        st.info("Add at least two ingested signs to see clusters.")
        return

    clustering = _cached_clustering(documents)
    max_clusters = min(len(documents), 30)
    default = min(4, max_clusters)
    cluster_count = st.slider(
        "Number of clusters",
        min_value=1,
        max_value=max_clusters,
        value=default,
        help="Cutting higher on the dendrogram merges clusters; lower splits them.",
    )
    cut_distance = _cut_distance(clustering, cluster_count)
    geometry = clustering.dendrogram()
    st.altair_chart(_dendrogram_chart(geometry, cut_distance), use_container_width=True)

    groups = clustering.cut(cluster_count)
    for index, group in enumerate(groups):
        st.markdown(f"### Cluster {index + 1} · {len(group.documents)} signs")
        thumbs = "".join(_cluster_thumb_html(services, document) for document in group.documents)
        if thumbs:
            st.markdown(
                f'<div style="display:flex;flex-wrap:wrap">{thumbs}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Assets could not be embedded inline; use the catalog tab to review.")
        theme_key = f"cluster-theme-{group.node_id}"
        if st.button("Summarize theme with LLM", key=f"btn-{theme_key}"):
            descriptions = tuple(document.text.search_text() for document in group.documents)
            with st.spinner("Asking cluster theme model…"):
                theme = _run(narrative.cluster_themes.summarize(descriptions))
            st.session_state[theme_key] = {"title": theme.title, "summary": theme.summary}
        theme_state = st.session_state.get(theme_key)
        if isinstance(theme_state, dict):
            typed_state = cast(dict[str, object], theme_state)
            title = str(typed_state.get("title", ""))
            summary = str(typed_state.get("summary", ""))
            st.success(f"**{title}** — {summary}")


@st.cache_data(show_spinner=False)
def _cached_clustering(
    documents: tuple[SearchDocument, ...],
) -> HierarchicalClustering:
    return average_linkage_cosine(documents)


def _cut_distance(clustering: HierarchicalClustering, cluster_count: int) -> float | None:
    if not clustering.merges or cluster_count <= 0:
        return None
    if cluster_count >= len(clustering.documents):
        return 0.0
    cut_index = len(clustering.documents) - cluster_count
    return clustering.merges[cut_index - 1].distance


def _select_anchor_and_companions(
    narrative: NarrativeServices,
    *,
    anchor_id: str,
    companion_count: int,
    similarity_mode: SimilarityMode,
    similarity_threshold: float,
    semantic_weight: float,
    statuses: frozenset[ReviewStatus],
    seed: int,
) -> NarrativeSelection:
    return narrative.selection.select(
        anchor_id,
        count=companion_count,
        maximum_similarity=similarity_threshold,
        statuses=statuses,
        mode=similarity_mode,
        semantic_weight=semantic_weight,
        seed=seed,
    )


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
        format_func=lambda value: _SIMILARITY_LABELS[SimilarityMode(value)],
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


def _anchor_options(
    narrative: NarrativeServices, statuses: frozenset[ReviewStatus]
) -> tuple[SignEmbeddingDocument, ...]:
    return narrative.selection.list_anchors(statuses)


def _narrative_tab(services: Services, narrative: NarrativeServices) -> None:
    session = st.session_state
    session.setdefault("narrative.system_prompt", default_system_prompt())
    session.setdefault("narrative.similarity_mode", SimilarityMode.HYBRID.value)
    session.setdefault("narrative.similarity_threshold", 0.55)
    session.setdefault("narrative.semantic_weight", 0.65)
    session.setdefault("narrative.anchor_weight", 2.5)

    statuses = DEFAULT_STATUSES
    anchors = _anchor_options(narrative, statuses)
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
            if st.button("Random anchor", use_container_width=True):
                seed = _new_seed()
                choice = narrative.selection.random_anchor(statuses, seed=seed)
                if choice is not None:
                    session["narrative.anchor_id"] = choice.item_id
                    st.rerun()

        settings = st.columns(4)
        with settings[0]:
            companion_count = st.slider(
                "Companion signs (n)",
                min_value=0,
                max_value=min(12, max(0, len(anchors) - 1)),
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
                format_func=lambda value: _LANGUAGE_LABELS[NarrativeLanguage(value)],
                index=[language.value for language in NarrativeLanguage].index(
                    str(session.get("narrative.language", NarrativeLanguage.ENGLISH.value))
                ),
            )
            session["narrative.language"] = language_value
        with settings[3]:
            strategy_value = st.selectbox(
                "Strategy",
                options=[strategy.value for strategy in NarrativeStrategy],
                format_func=lambda value: _STRATEGY_LABELS[NarrativeStrategy(value)],
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
                session["narrative.selection_seed"] = _new_seed()
        with actions[3]:
            generate = st.button("Generate story", type="primary")

    session.setdefault("narrative.selection_seed", _new_seed())
    selection = _select_anchor_and_companions(
        narrative,
        anchor_id=str(session["narrative.anchor_id"]),
        companion_count=int(cast(int, session["narrative.companion_count"])),
        similarity_mode=SimilarityMode(str(session["narrative.similarity_mode"])),
        similarity_threshold=float(cast(float, session["narrative.similarity_threshold"])),
        semantic_weight=float(cast(float, session["narrative.semantic_weight"])),
        statuses=statuses,
        seed=int(cast(int, session["narrative.selection_seed"])),
    )
    _render_selection_preview(services, selection)

    if generate:
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
                result = _run(narrative.experiments.generate(request, progress=reporter))
        except CancelCaptureError as error:
            st.error(f"Narrative generation failed: {error}")
        else:
            session["narrative.last_result"] = result.stored.relative_path
            st.success(f"Saved to `{result.stored.relative_path}`.")

    _render_last_result(narrative)
    st.divider()
    _render_saved_narratives(narrative)


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


def main() -> None:
    st.set_page_config(page_title="Cancel Capture", page_icon="🚫", layout="wide")
    st.title("Cancel Capture")
    st.caption("Private development interface · originals and metadata are not public")
    try:
        services = _core_services()
    except (CancelCaptureError, ValueError) as error:
        st.error(f"Configuration error: {error}")
        st.stop()

    tabs: Sequence[str] = ["Analyze", "Search", "Catalog", "Clusters", "Narrative"]
    analyze, search, catalog, clusters, narrative_tab = st.tabs(tabs)
    with analyze:
        _analyze_tab(services)
    with search:
        _search_tab(services)
    with catalog:
        _catalog_tab(services)

    narrative_services: NarrativeServices | None = None
    for target, label in ((clusters, "Clusters"), (narrative_tab, "Narrative")):
        with target:
            if narrative_services is None:
                try:
                    narrative_services = _narrative_services(services)
                except (CancelCaptureError, ValueError) as error:
                    st.error(f"Narrative provider not configured for the {label} tab: {error}")
                    continue
            if label == "Clusters":
                _clusters_tab(services, narrative_services)
            else:
                _narrative_tab(services, narrative_services)


if __name__ == "__main__":
    main()
