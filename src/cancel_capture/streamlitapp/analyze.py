from __future__ import annotations

import hashlib
import json
from typing import cast
from uuid import uuid4

import streamlit as st

from cancel_capture.container import Services
from cancel_capture.models import IngestionResult, IngestRequest, ReviewStatus, SourceKind
from cancel_capture.streamlitapp._shared import run


def render(services: Services) -> None:
    st.write(
        "Upload the same unprocessed image file that you would send to Telegram. The original "
        "is stored in the configured data volume."
    )
    uploaded = st.file_uploader(
        "Original image",
        type=["jpg", "jpeg", "png", "webp", "heic", "heif", "tif", "tiff"],
    )
    reanalyze = st.checkbox(
        "Reanalyze as a new run (uses provider API calls)",
        help="Enable when comparing prompt, model, or crop changes on the same reference file.",
    )
    if uploaded is not None and st.button("Analyze and crop", type="primary"):
        data = uploaded.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        source_key = f"streamlit:{digest}:{uuid4().hex}" if reanalyze else f"streamlit:{digest}"
        with st.status("Analyzing…", expanded=True) as status:
            result = run(
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
    st.subheader("Full photo")
    st.image(str(services.assets.resolve(result.original_relative_path)))
    st.write(result.description.en)
    metadata = cast(object, json.loads(result.metadata.raw_json))
    with st.expander("All extracted metadata"):
        st.json(metadata)

    st.subheader(f"Detected signs: {len(result.signs)}")
    if not result.signs:
        st.info("No qualifying prohibition sign was detected. The full photo is still archived.")
    for sign in result.signs:
        candidate = services.catalog.get_candidate(sign.item_id)
        left, right = st.columns([1, 2])
        with left:
            st.image(str(services.assets.resolve(candidate.crop_relative_path)))
            st.caption(f"Status: {candidate.status.value} · ID: {candidate.item_id}")
        with right:
            st.write(candidate.sign_description.en)
            st.write(f"Topics: {', '.join(candidate.topics_en)}")
            if candidate.status in (ReviewStatus.PENDING, ReviewStatus.FAILED) and st.button(
                "Reject locally",
                key=f"reject-{candidate.item_id}",
            ):
                services.review.reject(candidate.item_id, candidate.review_token, actor_user_id=0)
                st.rerun()
            st.caption("Channel publishing remains behind the Telegram confirmation button.")
