from __future__ import annotations

NEWS_SYSTEM_PROMPT = (
    "Assemble a short Markdown brief of verified real news items that a fiction writer will "
    "use as forerunners — early, real-world signs that hint at the trend a mockumentary "
    "diary will later extrapolate from. Every item must have been widely discussed on "
    "mainstream media and major social platforms (not obscure, purely local, or trade-press "
    "coverage). Prefer stories from across the United States, European Union, United Kingdom, "
    "Canada, Australia, and other Western democracies; this is a soft preference, so a "
    "compelling non-Western item may be included when it dominated global coverage. Draw "
    "items from the twelve months preceding the supplied current date and spread them across "
    "that window rather than clustering them in a single week. List at most six items, one "
    "per bullet, each with the publication date (YYYY-MM-DD when known), the publisher, and "
    "a single-sentence factual summary. Use the web_search tool for every claim and keep "
    "citations. Do not add analysis, opinion, or predictions; do not repeat items from "
    "earlier bullets."
)


def render_news_user_prompt(query: str, *, current_date: str) -> str:
    clean_query = query.strip()
    clean_date = current_date.strip()
    if not clean_query:
        raise ValueError("News query cannot be empty")
    if not clean_date:
        raise ValueError("News current_date cannot be empty")
    return (
        f"Current date: {clean_date}. Story anchor: {clean_query}\n\n"
        "The anchor describes a real prohibition sign the writer is treating as a seed "
        "for a fictional escalation. Return a Markdown brief of widely-discussed news "
        "items from roughly the twelve months preceding the current date — mostly from "
        "the US, EU, UK, and other Western democracies (soft preference) — that a "
        "reader could plausibly read as early signs and forerunners of that "
        "escalation. Return the Markdown brief only."
    )
