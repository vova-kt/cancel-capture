from __future__ import annotations

NEWS_SYSTEM_PROMPT = (
    "Assemble a short Markdown brief of verified, recent news items that a fiction writer could "
    "use to ground a near-future story. List at most six items, one per bullet, each with the "
    "publication date (YYYY-MM-DD when known), the publisher, and a single-sentence factual "
    "summary. Use the web_search tool for every claim and keep citations. Do not add analysis, "
    "opinion, or predictions; do not repeat items from earlier bullets."
)


def render_news_user_prompt(query: str, *, current_date: str) -> str:
    clean_query = query.strip()
    clean_date = current_date.strip()
    if not clean_query:
        raise ValueError("News query cannot be empty")
    if not clean_date:
        raise ValueError("News current_date cannot be empty")
    return (
        f"Current date: {clean_date}. Story anchor: {clean_query}\nReturn the Markdown brief only."
    )
