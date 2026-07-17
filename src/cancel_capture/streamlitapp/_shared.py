from __future__ import annotations

import asyncio
import base64
import secrets
from collections.abc import Awaitable
from pathlib import Path
from typing import Protocol, cast

import streamlit as st

from cancel_capture.application import SimilarityMode
from cancel_capture.config import AppConfig
from cancel_capture.container import (
    NarrativeServices,
    Services,
    build_narrative_services,
    build_services,
)
from cancel_capture.models import ReviewStatus
from cancel_capture.narrative_models import NarrativeLanguage
from cancel_capture.prompts import NarrativeStrategy

DEFAULT_STATUSES = frozenset({ReviewStatus.PUBLISHED, ReviewStatus.PENDING})

LANGUAGE_LABELS: dict[NarrativeLanguage, str] = {
    NarrativeLanguage.ENGLISH: "English",
    NarrativeLanguage.RUSSIAN: "Russian",
}
STRATEGY_LABELS: dict[NarrativeStrategy, str] = {
    NarrativeStrategy.FAMILY_CHRONICLE: "Family chronicle",
    NarrativeStrategy.CIVIC_RIPPLE: "Civic ripple",
    NarrativeStrategy.NEWS_MONTAGE: "News montage",
    NarrativeStrategy.EVERYDAY_ADAPTATION: "Everyday adaptation",
}
SIMILARITY_LABELS: dict[SimilarityMode, str] = {
    SimilarityMode.HYBRID: "Hybrid (semantic + visual)",
    SimilarityMode.SEMANTIC: "Semantic only",
    SimilarityMode.VISUAL: "Visual only",
}


@st.cache_resource(show_spinner=False)
def core_services() -> Services:
    return build_services(AppConfig.from_env())


@st.cache_resource(show_spinner=False)
def narrative_services() -> NarrativeServices:
    return build_narrative_services(core_services())


def new_seed() -> int:
    return secrets.randbits(31)


def run[T](coroutine: Awaitable[T]) -> T:
    return asyncio.run(_await(coroutine))


async def _await[T](coroutine: Awaitable[T]) -> T:
    return await coroutine


@st.cache_data(show_spinner=False, max_entries=512)
def image_to_data_uri(path_str: str, *, max_bytes: int = 512_000) -> str | None:
    path = Path(path_str)
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > max_bytes:
        return None
    suffix = path.suffix.lstrip(".").lower() or "jpeg"
    media_type = {"jpg": "jpeg", "jpe": "jpeg"}.get(suffix, suffix)
    return f"data:image/{media_type};base64,{base64.b64encode(data).decode('ascii')}"


class _StatusApi(Protocol):
    def update(self, *, label: str, state: str) -> None: ...

    def write(self, body: object) -> None: ...


class StreamlitProgress:
    """Bridge ProgressReporter onto an ``st.status`` container so ``note`` and ``stage`` calls
    made inside an ``asyncio.run`` inner loop reach the browser in near real time."""

    def __init__(self, status: object) -> None:
        self._status = cast("_StatusApi", status)

    def stage(self, key: str, label: str) -> None:
        self._status.update(label=label, state="running")

    def note(self, text: str) -> None:
        self._status.write(f"· {text}")

    def complete(self, label: str, *, ok: bool = True) -> None:
        self._status.update(label=label, state="complete" if ok else "error")
