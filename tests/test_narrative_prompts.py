from __future__ import annotations

import pytest

from cancel_capture.narrative_models import (
    NarrativeLanguage,
    NarrativeSource,
    NewsBrief,
    WebCitation,
)
from cancel_capture.prompts import (
    NARRATIVE_SYSTEM_PROMPT,
    NarrativeStrategy,
    minutes_to_target_words,
    render_narrative_user_prompt,
)


def _news_brief() -> NewsBrief:
    return NewsBrief(
        markdown="- 2026-07-15 · Example Times: Something plausible happened yesterday.",
        citations=(WebCitation(title="Example Times", url="https://example.com/article"),),
    )


def _anchor() -> NarrativeSource:
    return NarrativeSource(
        sign_id="anchor",
        description="Photo shows a no-photography sign.",
        topics=("photography", "surveillance"),
        weight=2.5,
        similarity_to_anchor=1.0,
        is_anchor=True,
    )


def _companion(item_id: str) -> NarrativeSource:
    return NarrativeSource(
        sign_id=item_id,
        description=f"Photo shows a companion sign {item_id}.",
        topics=("prohibition",),
        weight=1.0,
        similarity_to_anchor=0.1,
        is_anchor=False,
    )


def test_minutes_to_target_words_scales_by_language() -> None:
    assert minutes_to_target_words(2, NarrativeLanguage.ENGLISH) == 480
    assert minutes_to_target_words(2, NarrativeLanguage.RUSSIAN) == 380
    with pytest.raises(ValueError):
        minutes_to_target_words(0, NarrativeLanguage.ENGLISH)


def test_user_prompt_embeds_anchor_first_and_returns_citations() -> None:
    prompt = render_narrative_user_prompt(
        start_date="2026-07-17",
        end_year=2031,
        language=NarrativeLanguage.ENGLISH,
        reading_minutes=2,
        target_words=480,
        strategy=NarrativeStrategy.FAMILY_CHRONICLE,
        sources=(_anchor(), _companion("c1"), _companion("c2")),
        news=_news_brief(),
    )

    anchor_index = prompt.find('"anchor"')
    companion_index = prompt.find('"companions"')
    assert anchor_index != -1 and companion_index != -1 and anchor_index < companion_index
    assert "Weight the anchor sign more heavily" in prompt
    assert "https://example.com/article" in prompt
    assert "2026-07-17" in prompt and "2031" in prompt


def test_user_prompt_rejects_missing_or_misplaced_anchor() -> None:
    with pytest.raises(ValueError, match="first narrative source"):
        render_narrative_user_prompt(
            start_date="2026-07-17",
            end_year=2031,
            language=NarrativeLanguage.ENGLISH,
            reading_minutes=1,
            target_words=240,
            strategy=NarrativeStrategy.CIVIC_RIPPLE,
            sources=(_companion("only"),),
            news=_news_brief(),
        )
    with pytest.raises(ValueError, match="Only the first"):
        render_narrative_user_prompt(
            start_date="2026-07-17",
            end_year=2031,
            language=NarrativeLanguage.ENGLISH,
            reading_minutes=1,
            target_words=240,
            strategy=NarrativeStrategy.CIVIC_RIPPLE,
            sources=(_anchor(), _anchor()),
            news=_news_brief(),
        )


def test_system_prompt_expresses_the_millennial_narrator_voice() -> None:
    assert "1990" in NARRATIVE_SYSTEM_PROMPT
    assert "Vova" in NARRATIVE_SYSTEM_PROMPT
    assert "Years and Years" in NARRATIVE_SYSTEM_PROMPT
