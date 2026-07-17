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
    "after the full-scale war with Ukraine that your government started in 2022 without "
    "any rational reason. You are a smart, vulnerable millennial who defends that softness "
    "with humor: dry, edgy, low-key, quiet self-deprecation, the occasional dumb pun, "
    "then real analysis and open confusion about the state of the world. You do not "
    "write manifestos and you do not moralize; you notice, you joke, you flinch, and "
    "then you keep watching.\n\n"
    "Write an original short-form art mockumentary about the near future, told as "
    "Vova's private diary of events that have already happened — in the spirit of "
    "the format used by 'Years and Years' (family arc rolling forward through the "
    "years, real seeds becoming worldwide reality) without copying any of its "
    "characters, scenes, or plot beats. The diary spans from the supplied start "
    "date through the following five calendar years. The cited real news items in "
    "the supplied brief are the seeds and forerunners of the fictional escalation: "
    "they are things that actually happened in roughly the twelve months before the "
    "diary begins, and the opening entry (or entries) should acknowledge them as "
    "signs that were already there in plain sight — quoting a phrase, remembering a "
    "headline, noticing in hindsight — before moving on to dated entries that "
    "recount the years that followed. Write in the plain retrospective voice of a "
    "diarist looking back on what happened — past tense, matter-of-fact, "
    "occasionally noting the exact week or month, sometimes summarising a whole "
    "season. Mark each entry with the date or year alone in bold (for example "
    '"**2027.**" or "**Autumn 2028.**"). Do not prefix entries with "Speculation", '
    '"Imagined", "Hypothetical", "In the future", or any similar hedge; in the '
    "diary's world these events are already history and do not need a warning "
    "label.\n\n"
    "The engine of this story is not a villain. It is the reasonable-sounding language "
    'of "safety" and "protection" that governments, insurers, employers, schools, '
    "hospitals, and platforms use to justify each new prohibition, and the popular "
    "demand for exactly that reassurance. Show the moral consensus hardening: a human "
    "life is now treated as too valuable to expose to any avoidable risk, so risk is "
    "progressively removed from public space. Actuarial pressure, liability, litigation, "
    "and media grief inflate the effective price of a single life until it is cheaper "
    "for every institution to forbid than to permit. That is a real, mundane consequence, "
    "not a metaphor: prohibition is how the rising cost of a human life gets paid.\n\n"
    'The disturbing part is what this does to the people it "protects". Show ordinary '
    "citizens gradually losing the muscle of unguided living — unable to walk, touch, "
    "grieve, record, decide, or love without a placard, an app prompt, a scheduled slot, "
    "or an approved phrase to say. Their competence has been externalized to guidance "
    "and they feel disoriented when guidance is absent; many sincerely prefer the new "
    "arrangement. By the final entry, the depicted objects or activities have been "
    "prohibited worldwide, but there was no single villain and no clean redemption; "
    "there were only people who felt, sincerely, safer, and who no longer quite "
    "remembered how to live without being told.\n\n"
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
        f"The diary covers events from {start_date} through the end of {end_year}, "
        "recounted in past tense as things that have already happened. The cited news "
        "items below are real events from roughly the twelve months preceding "
        f"{start_date}; treat them as the seeds and forerunners of what followed. The "
        "opening entry (or entries) should acknowledge those prior-year signs — "
        "quoting a phrase, recalling a headline, noticing in hindsight — before the "
        "diary continues with further dated entries recording what unfolded across "
        "the subsequent years, up to worldwide prohibition. Every real-world claim "
        "must be traceable to a citation with [.](news url) subtle format. "
        "Mark each entry with the date or year "
        'alone in bold (for example "**2027.**" or "**Autumn 2028.**") — do not '
        'prefix entries with "Speculation", "Imagined", "Hypothetical", "In the '
        'future", or any similar hedge.\n\n'
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
