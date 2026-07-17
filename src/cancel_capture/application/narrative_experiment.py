from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from random import Random

from cancel_capture.application.narrative_selection import (
    NarrativeSelection,
    SelectedNarrativeSign,
    SimilarityMode,
)
from cancel_capture.narrative_models import (
    NarrativeArtifact,
    NarrativeArtifactMetadata,
    NarrativeGenerationRequest,
    NarrativeLanguage,
    NarrativeSource,
    NewsBrief,
    StoredNarrativeArtifact,
)
from cancel_capture.ports import CurrentNewsProvider, NarrativeProvider, NarrativeStore
from cancel_capture.progress import NullProgress, ProgressReporter, with_periodic_notes
from cancel_capture.prompts import (
    NARRATIVE_SYSTEM_PROMPT,
    NarrativeStrategy,
    minutes_to_target_words,
    render_narrative_user_prompt,
)
from cancel_capture.wait_lines import random_wait_line

DEFAULT_ANCHOR_WEIGHT = 2.5
DEFAULT_COMPANION_WEIGHT = 1.0
DEFAULT_WAIT_INTERVAL_SECONDS = 6.0


@dataclass(frozen=True, slots=True)
class NarrativeExperimentRequest:
    selection: NarrativeSelection
    strategy: NarrativeStrategy
    language: NarrativeLanguage
    reading_minutes: int
    system_prompt: str
    similarity_mode: SimilarityMode
    similarity_threshold: float
    semantic_weight: float
    random_seed: int | None
    news_query: str
    start_date: str | None = None
    end_year: int | None = None
    anchor_weight: float = DEFAULT_ANCHOR_WEIGHT
    companion_weight: float = DEFAULT_COMPANION_WEIGHT

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError("Narrative system prompt cannot be empty")
        if not self.news_query.strip():
            raise ValueError("Narrative news query cannot be empty")
        if self.reading_minutes <= 0:
            raise ValueError("Narrative reading minutes must be positive")
        if self.anchor_weight <= 0 or self.companion_weight <= 0:
            raise ValueError("Narrative anchor and companion weights must be positive")
        if self.anchor_weight < self.companion_weight:
            raise ValueError("Anchor weight must be at least the companion weight")


@dataclass(frozen=True, slots=True)
class NarrativeExperimentResult:
    stored: StoredNarrativeArtifact
    news: NewsBrief
    request: NarrativeGenerationRequest


class NarrativeExperimentService:
    def __init__(
        self,
        news: CurrentNewsProvider,
        narrative: NarrativeProvider,
        store: NarrativeStore,
        *,
        wait_interval_seconds: float = DEFAULT_WAIT_INTERVAL_SECONDS,
    ) -> None:
        self._news = news
        self._narrative = narrative
        self._store = store
        self._wait_interval_seconds = wait_interval_seconds

    async def generate(
        self,
        experiment: NarrativeExperimentRequest,
        *,
        progress: ProgressReporter | None = None,
    ) -> NarrativeExperimentResult:
        reporter: ProgressReporter = progress or NullProgress()
        rng = Random()  # entertainment lines are always fresh, unrelated to sampling seed
        now = datetime.now(UTC)
        start_date = experiment.start_date or now.date().isoformat()
        end_year = experiment.end_year or (now.year + 5)
        target_words = minutes_to_target_words(experiment.reading_minutes, experiment.language)

        reporter.stage("news", "Reading current events")
        reporter.note(random_wait_line("news", rng=rng))
        news = await with_periodic_notes(
            self._news.research(experiment.news_query, current_date=start_date),
            note_provider=lambda: random_wait_line("news", rng=rng),
            interval_seconds=self._wait_interval_seconds,
            reporter=reporter,
        )

        sources = _build_sources(experiment)
        user_prompt = render_narrative_user_prompt(
            start_date=start_date,
            end_year=end_year,
            language=experiment.language,
            reading_minutes=experiment.reading_minutes,
            target_words=target_words,
            strategy=experiment.strategy,
            sources=sources,
            news=news,
        )
        generation_request = NarrativeGenerationRequest(
            start_date=start_date,
            end_year=end_year,
            language=experiment.language,
            reading_minutes=experiment.reading_minutes,
            target_words=target_words,
            strategy=experiment.strategy.value,
            system_prompt=experiment.system_prompt,
            user_prompt=user_prompt,
            news=news,
            sources=sources,
        )

        reporter.stage("drafting", "Writing the story")
        reporter.note(random_wait_line("drafting", rng=rng))
        draft = await with_periodic_notes(
            self._narrative.generate(generation_request),
            note_provider=lambda: random_wait_line("drafting", rng=rng),
            interval_seconds=self._wait_interval_seconds,
            reporter=reporter,
        )

        reporter.stage("saving", "Filing the story")
        reporter.note(random_wait_line("saving", rng=rng))
        identity = self._narrative.identity
        metadata = NarrativeArtifactMetadata.create(
            anchor_sign_id=experiment.selection.anchor.document.item_id,
            source_sign_ids=tuple(
                companion.document.item_id for companion in experiment.selection.companions
            ),
            strategy=experiment.strategy.value,
            language=experiment.language.value,
            reading_minutes=experiment.reading_minutes,
            similarity_mode=experiment.similarity_mode.value,
            similarity_threshold=experiment.similarity_threshold,
            semantic_weight=experiment.semantic_weight,
            random_seed=experiment.random_seed,
            provider=identity.provider,
            model=identity.model,
            provider_namespace=identity.namespace,
            system_prompt=experiment.system_prompt,
            user_prompt=user_prompt,
            web_citations=news.citations,
            created_at=now,
        )
        artifact = NarrativeArtifact(
            title=draft.title,
            description=draft.description,
            body_markdown=draft.body_markdown,
            metadata=metadata,
        )
        stored = self._store.save(artifact)
        reporter.complete("Story ready", ok=True)
        return NarrativeExperimentResult(stored=stored, news=news, request=generation_request)

    def list_saved(self) -> tuple[StoredNarrativeArtifact, ...]:
        return self._store.list_artifacts()

    def read_saved(self, relative_path: str) -> StoredNarrativeArtifact:
        return self._store.read(relative_path)


def default_system_prompt() -> str:
    return NARRATIVE_SYSTEM_PROMPT


def _build_sources(experiment: NarrativeExperimentRequest) -> tuple[NarrativeSource, ...]:
    anchor = _source_from_selected(experiment.selection.anchor, experiment.anchor_weight)
    companions = tuple(
        _source_from_selected(companion, experiment.companion_weight)
        for companion in experiment.selection.companions
    )
    return (anchor, *companions)


def _source_from_selected(selected: SelectedNarrativeSign, weight: float) -> NarrativeSource:
    document = selected.document
    topics = tuple(dict.fromkeys((*document.topics_en, *document.topics_ru)))
    return NarrativeSource(
        sign_id=document.item_id,
        description=document.text.search_text(),
        topics=topics,
        weight=weight,
        similarity_to_anchor=selected.similarity_to_anchor,
        is_anchor=selected.is_anchor,
    )
