from __future__ import annotations

import asyncio
import hashlib
import json
from typing import cast
from uuid import uuid4

import streamlit as st

from cancel_capture.config import AppConfig
from cancel_capture.container import Services, build_services
from cancel_capture.errors import CancelCaptureError
from cancel_capture.models import (
    IngestionResult,
    IngestRequest,
    ItemKind,
    ReviewStatus,
    SourceKind,
)


def _services() -> Services:
    return build_services(AppConfig.from_env())


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
            result = asyncio.run(
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


def _search_tab(services: Services) -> None:
    query = st.text_input("Topic in English or Russian / Тема на английском или русском")
    scope = st.selectbox("Scope", ["all", ItemKind.SIGN.value, ItemKind.PHOTO.value])
    if query.strip() and st.button("Search / Искать"):
        kind = None if scope == "all" else ItemKind(scope)
        hits = asyncio.run(services.search.search(query, kind=kind, limit=30))
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


def main() -> None:
    st.set_page_config(page_title="Cancel Capture", page_icon="🚫", layout="wide")
    st.title("Cancel Capture")
    st.caption("Private development interface · originals and metadata are not public")
    try:
        services = _services()
    except (CancelCaptureError, ValueError) as error:
        st.error(f"Configuration error: {error}")
        st.stop()
    analyze, search, catalog = st.tabs(["Analyze", "Search", "Catalog"])
    with analyze:
        _analyze_tab(services)
    with search:
        _search_tab(services)
    with catalog:
        _catalog_tab(services)


if __name__ == "__main__":
    main()
