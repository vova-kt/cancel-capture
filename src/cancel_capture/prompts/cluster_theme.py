from __future__ import annotations

import json

CLUSTER_THEME_SYSTEM_PROMPT = (
    "Label a cluster of prohibition-sign descriptions for a private visual archive. "
    "Identify only the concrete theme shared by most signs. Do not infer a political "
    "trend, location, motive, or fact that is absent from the descriptions. Write a "
    "specific title of at most six words and one concise explanatory sentence."
)


def render_cluster_theme_user_prompt(descriptions: tuple[str, ...]) -> str:
    clean_descriptions = tuple(description.strip() for description in descriptions)
    if not clean_descriptions or any(not description for description in clean_descriptions):
        raise ValueError("Cluster theme generation requires non-empty descriptions")
    payload = json.dumps(clean_descriptions, ensure_ascii=False, separators=(",", ":"))
    return f"Describe the shared theme of these sign descriptions:\n{payload}"
