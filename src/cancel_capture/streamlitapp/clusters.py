from __future__ import annotations

import html
from typing import cast

import altair as alt
import pandas as pd  # pyright: ignore [reportMissingTypeStubs]
import streamlit as st

from cancel_capture.application.clustering import (
    DendrogramGeometry,
    HierarchicalClustering,
    average_linkage_cosine,
)
from cancel_capture.container import NarrativeServices, Services
from cancel_capture.models import ItemKind, SearchDocument
from cancel_capture.streamlitapp._shared import core_services, image_to_data_uri, run


def render(services: Services, narrative: NarrativeServices) -> None:
    documents = _sign_documents_cached()
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
    st.altair_chart(_dendrogram_chart(geometry, cut_distance), width="stretch")

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
                theme = run(narrative.cluster_themes.summarize(descriptions))
            st.session_state[theme_key] = {"title": theme.title, "summary": theme.summary}
        theme_state = st.session_state.get(theme_key)
        if isinstance(theme_state, dict):
            typed_state = cast(dict[str, object], theme_state)
            title = str(typed_state.get("title", ""))
            summary = str(typed_state.get("summary", ""))
            st.success(f"**{title}** — {summary}")


@st.cache_data(show_spinner=False, ttl=30)
def _sign_documents_cached() -> tuple[SearchDocument, ...]:
    services = core_services()
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


def _cluster_thumb_html(services: Services, document: SearchDocument) -> str:
    data_uri = image_to_data_uri(str(services.assets.resolve(document.asset_relative_path)))
    if data_uri is None:
        return ""
    tooltip = html.escape(document.text.en, quote=True)
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
