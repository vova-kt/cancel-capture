from __future__ import annotations

from pathlib import Path

from cancel_capture.adapters.markdown_narratives import MarkdownNarrativeStore
from cancel_capture.application.narrative_experiment import (
    NarrativeExperimentRequest,
    NarrativeExperimentService,
    default_system_prompt,
)
from cancel_capture.application.narrative_selection import (
    NarrativeSelection,
    SelectedNarrativeSign,
    SimilarityMode,
)
from cancel_capture.models import (
    BilingualText,
    Embedding,
    ProviderIdentity,
    ReviewStatus,
    SignEmbeddingDocument,
)
from cancel_capture.narrative_models import (
    NarrativeDraft,
    NarrativeGenerationRequest,
    NarrativeLanguage,
    NewsBrief,
    WebCitation,
)
from cancel_capture.prompts import NarrativeStrategy

SEMANTIC_IDENTITY = ProviderIdentity("test", "semantic-v1")
NARRATIVE_IDENTITY = ProviderIdentity("openai", "gpt-5.6-terra", "test-namespace")


def _document(item_id: str) -> SignEmbeddingDocument:
    return SignEmbeddingDocument(
        item_id=item_id,
        parent_photo_id=f"photo-{item_id}",
        text=BilingualText(en=f"English {item_id}", ru=f"Russian {item_id}"),
        topics_en=("prohibition",),
        topics_ru=("запрет",),
        asset_relative_path=f"assets/crops/{item_id}.jpg",
        status=ReviewStatus.PUBLISHED,
        semantic_embedding=Embedding(identity=SEMANTIC_IDENTITY, values=(1.0, 0.0)),
        visual_embedding=None,
    )


def _selection() -> NarrativeSelection:
    anchor = SelectedNarrativeSign(_document("anchor"), similarity_to_anchor=1.0, is_anchor=True)
    companion = SelectedNarrativeSign(_document("companion"), similarity_to_anchor=0.1)
    return NarrativeSelection(
        anchor=anchor,
        companions=(companion,),
        eligible_count=1,
        requested_count=1,
    )


class FakeCurrentNews:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def identity(self) -> ProviderIdentity:
        return ProviderIdentity("openai", "gpt-5.6-terra", "test-namespace")

    async def research(self, query: str, *, current_date: str) -> NewsBrief:
        self.calls += 1
        return NewsBrief(
            markdown=f"- Recent event about {query} on {current_date}",
            citations=(WebCitation(title="Test News", url="https://example.com/a"),),
        )


class FakeNarrative:
    def __init__(self) -> None:
        self.requests: list[NarrativeGenerationRequest] = []

    @property
    def identity(self) -> ProviderIdentity:
        return NARRATIVE_IDENTITY

    async def generate(self, request: NarrativeGenerationRequest) -> NarrativeDraft:
        self.requests.append(request)
        return NarrativeDraft(
            title=f"Fictional {request.language.value} title",
            description="A short synthetic description.",
            body_markdown="# Body\n\nA short synthetic body.",
        )


async def test_experiment_saves_narrative_and_records_metadata(tmp_path: Path) -> None:
    store = MarkdownNarrativeStore(tmp_path)
    news = FakeCurrentNews()
    narrative = FakeNarrative()
    service = NarrativeExperimentService(news=news, narrative=narrative, store=store)

    request = NarrativeExperimentRequest(
        selection=_selection(),
        strategy=NarrativeStrategy.FAMILY_CHRONICLE,
        language=NarrativeLanguage.ENGLISH,
        reading_minutes=2,
        system_prompt=default_system_prompt(),
        similarity_mode=SimilarityMode.SEMANTIC,
        similarity_threshold=0.55,
        semantic_weight=0.65,
        random_seed=17,
        news_query="Anchor topic",
    )
    result = await service.generate(request)

    assert news.calls == 1
    assert len(narrative.requests) == 1
    generation = narrative.requests[0]
    assert generation.sources[0].sign_id == "anchor"
    assert generation.sources[0].weight >= generation.sources[1].weight

    stored = result.stored
    assert stored.artifact.metadata.anchor_sign_id == "anchor"
    assert stored.artifact.metadata.source_sign_ids == ("companion",)
    assert stored.artifact.metadata.web_citations
    reread = service.read_saved(stored.relative_path)
    assert reread == stored
    assert service.list_saved() == (stored,)


async def test_experiment_request_validates_weights_and_prompt() -> None:
    import pytest

    with pytest.raises(ValueError, match="system prompt"):
        NarrativeExperimentRequest(
            selection=_selection(),
            strategy=NarrativeStrategy.CIVIC_RIPPLE,
            language=NarrativeLanguage.ENGLISH,
            reading_minutes=2,
            system_prompt=" ",
            similarity_mode=SimilarityMode.HYBRID,
            similarity_threshold=0.5,
            semantic_weight=0.5,
            random_seed=0,
            news_query="topic",
        )

    with pytest.raises(ValueError, match="at least the companion"):
        NarrativeExperimentRequest(
            selection=_selection(),
            strategy=NarrativeStrategy.CIVIC_RIPPLE,
            language=NarrativeLanguage.ENGLISH,
            reading_minutes=2,
            system_prompt="A prompt",
            similarity_mode=SimilarityMode.HYBRID,
            similarity_threshold=0.5,
            semantic_weight=0.5,
            random_seed=0,
            news_query="topic",
            anchor_weight=0.5,
            companion_weight=1.0,
        )
