from __future__ import annotations

import json
from enum import StrEnum


class NarrativeStrategy(StrEnum):
    FAMILY_CHRONICLE = "family_chronicle"
    CIVIC_RIPPLE = "civic_ripple"
    NEWS_MONTAGE = "news_montage"
    EVERYDAY_ADAPTATION = "everyday_adaptation"


NARRATIVE_STRATEGY_PROMPTS: dict[NarrativeStrategy, str] = {
    NarrativeStrategy.FAMILY_CHRONICLE: (
        "Follow one ordinary extended family whose disagreements make each new ban personal. "
        "Let public change arrive through work, school, care, love, and small domestic compromises."
    ),
    NarrativeStrategy.CIVIC_RIPPLE: (
        "Begin with one plausible local rule, then follow its legal, commercial, technological, "
        "and social aftershocks as similar bans spread between countries."
    ),
    NarrativeStrategy.NEWS_MONTAGE: (
        "Interleave intimate scenes with brief broadcasts, messages, product notices, and official "
        "announcements. The fragments should reveal normalization without becoming an exposition dump."
    ),
    NarrativeStrategy.EVERYDAY_ADAPTATION: (
        "Focus on the improvised habits, euphemisms, black markets, jokes, and quiet losses through "
        "which ordinary people adapt as the prohibited things disappear from public life."
    ),
}

NARRATIVE_SYSTEM_PROMPT = (
    "Write an original near-future ensemble social drama that unfolds across five years. "
    "Combine intimate domestic consequences with rapidly changing politics, technology, media, "
    "and public language. Escalation should feel plausible: extraordinary restrictions become "
    "ordinary through convenience, fear, bureaucracy, commerce, and repetition. Use precise lived "
    "details, restrained dark humor, moral ambiguity, and time jumps that show cumulative change. "
    "The supplied prohibition signs are documentary seeds, not proof of a real trend; transform them "
    "into clearly speculative fiction. By the final year, the depicted activities or objects are "
    "banned worldwide, but avoid a simplistic manifesto or a single all-powerful villain. Do not copy "
    "characters, dialogue, scenes, or plotlines from any existing television series. Return a short "
    "title, a one-sentence description, and a polished standalone story."
)


def render_narrative_user_prompt(
    *,
    anchor_description: str,
    reference_descriptions: tuple[str, ...],
    strategy: NarrativeStrategy,
    start_year: int,
) -> str:
    anchor = anchor_description.strip()
    references = tuple(description.strip() for description in reference_descriptions)
    if not anchor:
        raise ValueError("Narrative generation requires an anchor description")
    if not references or any(not description for description in references):
        raise ValueError("Narrative generation requires non-empty reference descriptions")
    if start_year < 1:
        raise ValueError("Narrative start year must be positive")
    evidence = json.dumps(
        {"anchor": anchor, "references": references},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    strategy_prompt = NARRATIVE_STRATEGY_PROMPTS[strategy]
    return (
        f"The story runs from {start_year} through {start_year + 5}.\n"
        f"Narrative approach: {strategy_prompt}\n"
        "Use every supplied sign meaningfully, while keeping all real-world claims factual and all "
        f"future developments explicitly fictional. Source descriptions:\n{evidence}"
    )
