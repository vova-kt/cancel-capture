from __future__ import annotations

import streamlit as st

from cancel_capture.container import Services
from cancel_capture.models import ItemKind
from cancel_capture.streamlitapp._shared import run


def render(services: Services) -> None:
    query = st.text_input("Topic")
    scope = st.selectbox("Scope", ["all", ItemKind.SIGN.value, ItemKind.PHOTO.value])
    if query.strip() and st.button("Search"):
        kind = None if scope == "all" else ItemKind(scope)
        hits = run(services.search.search(query, kind=kind, limit=30))
        st.caption(f"{len(hits)} results")
        for hit in hits:
            left, right = st.columns([1, 3])
            with left:
                st.image(str(services.assets.resolve(hit.asset_relative_path)))
            with right:
                st.write(f"**{hit.score:.3f} · {hit.kind.value} · {hit.status.value}**")
                st.write(hit.description.en)
