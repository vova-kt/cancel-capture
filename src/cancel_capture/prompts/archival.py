from __future__ import annotations

import json
from collections.abc import Mapping


ARCHIVAL_TEXT_SYSTEM_PROMPT = (
    "Write accurate archival descriptions in both English and Russian. The full "
    "photo description must cover setting, objects, people, composition, and the "
    "relationship of signs to the scene without guessing hidden facts. Each sign "
    "description must explain exactly what appears prohibited, its pictogram, "
    "wording, condition, and visual context. Preserve quoted text. Produce concise "
    "topic tags in both languages that will make semantic retrieval useful. Do not "
    "write a narrative about society or infer motivations. Keep sign ordinals "
    "unchanged."
)


def render_archival_text_user_prompt(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
