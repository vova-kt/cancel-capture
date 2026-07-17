from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from cancel_capture.container import NarrativeServices
from cancel_capture.errors import CancelCaptureError
from cancel_capture.streamlitapp import analyze, catalog, clusters, narrative, search
from cancel_capture.streamlitapp._shared import core_services, narrative_services


def main() -> None:
    st.set_page_config(page_title="Cancel Capture", page_icon="🚫", layout="wide")
    st.title("Cancel Capture")
    st.caption("Private development interface · originals and metadata are not public")
    try:
        services = core_services()
    except (CancelCaptureError, ValueError) as error:
        st.error(f"Configuration error: {error}")
        st.stop()

    tabs: Sequence[str] = ["Analyze", "Search", "Catalog", "Clusters", "Narrative"]
    analyze_tab, search_tab, catalog_tab, clusters_tab, narrative_tab = st.tabs(tabs)
    with analyze_tab:
        analyze.render(services)
    with search_tab:
        search.render(services)
    with catalog_tab:
        catalog.render(services)

    narrative_bundle: NarrativeServices | None = None
    for target, label, tab_render in (
        (clusters_tab, "Clusters", clusters.render),
        (narrative_tab, "Narrative", narrative.render),
    ):
        with target:
            if narrative_bundle is None:
                try:
                    narrative_bundle = narrative_services()
                except (CancelCaptureError, ValueError) as error:
                    st.error(f"Narrative provider not configured for the {label} tab: {error}")
                    continue
            tab_render(services, narrative_bundle)


if __name__ == "__main__":
    main()
