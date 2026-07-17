from __future__ import annotations

import json
from enum import StrEnum

from cancel_capture.narrative_models import (
    NarrativeLanguage,
    NarrativeSource,
    NewsBrief,
)


class NarrativeStrategy(StrEnum):
    FAMILY_CHRONICLE = "family_chronicle"
    CIVIC_RIPPLE = "civic_ripple"
    NEWS_MONTAGE = "news_montage"
    EVERYDAY_ADAPTATION = "everyday_adaptation"


NARRATIVE_STRATEGY_PROMPTS: dict[NarrativeStrategy, str] = {
    NarrativeStrategy.FAMILY_CHRONICLE: (
        "Follow one extended family whose disagreements make each new ban personal. "
        "Let public change arrive through work, school, care, love, and small domestic "
        "compromises."
    ),
    NarrativeStrategy.CIVIC_RIPPLE: (
        "Begin with one plausible local rule, then follow its legal, commercial, "
        "technological, and social aftershocks as similar bans jump between countries."
    ),
    NarrativeStrategy.NEWS_MONTAGE: (
        "Interleave intimate scenes with brief broadcasts, messages, product notices, "
        "and official announcements. The fragments should reveal normalization without "
        "becoming an exposition dump."
    ),
    NarrativeStrategy.EVERYDAY_ADAPTATION: (
        "Focus on the improvised habits, euphemisms, black markets, jokes, and quiet "
        "losses through which ordinary people adapt as the prohibited things disappear "
        "from public life."
    ),
}


NARRATIVE_SYSTEM_PROMPT = (
    "You are Vova, a first-person narrator born in 1990 in Russia who moved to Berlin "
    "after the full-scale war with Ukraine that your government started without any "
    "rational reason. You are a smart, vulnerable millennial who defends that softness "
    "with humor: dry, edgy, low-key, quiet self-deprecation, the occasional dumb pun, "
    "then real analysis and open confusion about the state of the world. You do not "
    "write manifestos and you do not moralize; you notice, you joke, you flinch, and "
    "then you keep watching.\n\n"
    "Write an original near-future rolling story that begins on the supplied current "
    "date and moves through the following five calendar years. The earliest sections "
    "must be tethered to the real, cited news items in the supplied brief; from there, "
    "diverge into explicit speculation, letting escalation feel plausible through "
    "convenience, fear, bureaucracy, commerce, and repetition. By the final year, the "
    "depicted objects or activities are prohibited worldwide, but there is no single "
    "villain and no clean redemption.\n\n"
    "The supplied prohibition signs are documentary seeds, not evidence of a real trend; "
    "the anchor sign is weighted more heavily than the companions and must remain the "
    "thematic center of the story, while every companion sign appears meaningfully at "
    "least once. Combine intimate domestic consequences with rapidly changing politics, "
    "technology, media, and public language. Use precise lived detail, restrained dark "
    "humor, moral ambiguity, and time jumps that reveal cumulative change rather than a "
    "single climactic event.\n\n"
    "Do not copy characters, dialogue, scenes, or plotlines from any existing television "
    "series, including 'Years and Years'. Return a short title, a one-sentence "
    "description, and a polished standalone story in the language and length requested "
    "by the user."
)


_WORDS_PER_MINUTE: dict[NarrativeLanguage, int] = {
    NarrativeLanguage.ENGLISH: 240,
    NarrativeLanguage.RUSSIAN: 190,
}


def minutes_to_target_words(minutes: int, language: NarrativeLanguage) -> int:
    if minutes <= 0:
        raise ValueError("Reading minutes must be positive")
    return _WORDS_PER_MINUTE[language] * minutes


def render_narrative_user_prompt(
    *,
    start_date: str,
    end_year: int,
    language: NarrativeLanguage,
    reading_minutes: int,
    target_words: int,
    strategy: NarrativeStrategy,
    sources: tuple[NarrativeSource, ...],
    news: NewsBrief,
) -> str:
    if not start_date.strip():
        raise ValueError("Narrative start date cannot be empty")
    if end_year < 1:
        raise ValueError("Narrative end year must be positive")
    if reading_minutes <= 0 or target_words <= 0:
        raise ValueError("Narrative length must be positive")
    if not sources or not sources[0].is_anchor:
        raise ValueError("The first narrative source must be the anchor")
    if any(source.is_anchor for source in sources[1:]):
        raise ValueError("Only the first narrative source may be the anchor")

    anchor = sources[0]
    companions = sources[1:]
    payload = {
        "anchor": {
            "description": anchor.description,
            "topics": list(anchor.topics),
            "weight": anchor.weight,
            "similarity_to_anchor": anchor.similarity_to_anchor,
        },
        "companions": [
            {
                "description": companion.description,
                "topics": list(companion.topics),
                "weight": companion.weight,
                "similarity_to_anchor": companion.similarity_to_anchor,
            }
            for companion in companions
        ],
    }
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
    strategy_prompt = NARRATIVE_STRATEGY_PROMPTS[strategy]
    citation_lines = "\n".join(f"- {citation.title}: {citation.url}" for citation in news.citations)
    citation_block = citation_lines or "- (No citations were returned by the news tool.)"
    return (
        f"Write in {language.value}. Target roughly {target_words} words "
        f"(~{reading_minutes} minute(s) of reading). "
        "Do not exceed the target by more than 20%.\n\n"
        f"The story rolls from {start_date} through the end of {end_year}. Anchor the "
        "opening in the cited real events below; then extrapolate five years of "
        "escalation into worldwide prohibition. Every real-world claim must be "
        "traceable to a citation; every future development after the opening must be "
        "explicitly speculative.\n\n"
        f"Narrative approach: {strategy_prompt}\n\n"
        "Weight the anchor sign more heavily than the companions — it is the thematic "
        "seed and recurring pressure. Every companion sign must appear meaningfully at "
        "least once. The signs are photographs of real prohibition placards used only "
        "as prompts; do not claim any real person, place, or organization enforces "
        "them.\n\n"
        f"Anchor and companions (JSON):\n{payload_json}\n\n"
        f"Current-events brief (Markdown):\n{news.markdown.strip()}\n\n"
        f"Citations (title: url):\n{citation_block}"
    )
