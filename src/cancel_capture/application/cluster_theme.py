from __future__ import annotations

from cancel_capture.narrative_models import ClusterTheme
from cancel_capture.ports import ClusterThemeProvider
from cancel_capture.prompts import (
    CLUSTER_THEME_SYSTEM_PROMPT,
    render_cluster_theme_user_prompt,
)


class ClusterThemeService:
    def __init__(self, provider: ClusterThemeProvider) -> None:
        self._provider = provider

    async def summarize(self, descriptions: tuple[str, ...]) -> ClusterTheme:
        user_prompt = render_cluster_theme_user_prompt(descriptions)
        return await self._provider.summarize(CLUSTER_THEME_SYSTEM_PROMPT, user_prompt)
