# Streamlit-only experiments

The Streamlit UI adds two experiment tabs on top of the ingestion pipeline. They live behind an
optional narrative-provider service graph
([`build_narrative_services`](../src/cancel_capture/container.py)) that is composed lazily so the
bot and CLI never need a narrative API key.

## Clustering

- Hierarchical clustering runs in-process with a deterministic average-linkage cosine
  implementation
  ([`average_linkage_cosine`](../src/cancel_capture/application/clustering.py)). Ties break on
  sorted member sets so the same inputs always produce the same tree. All merges plus dendrogram
  segment coordinates are computed once per session; the cut is a pure function of the slider.
- Semantic embeddings are read via
  [`SQLiteCatalog.list_sign_embedding_documents`](../src/cancel_capture/adapters/sqlite_catalog.py)
  and converted into `SearchDocument`s so the same clustering surface works for future embedding
  variants. Visual embeddings live in a separate `visual_embeddings` table (schema migration 7),
  refreshed by
  [`VisualEmbeddingService.ensure_current`](../src/cancel_capture/application/visual_embeddings.py)
  through the deterministic
  [`PillowVisualEmbeddingProvider`](../src/cancel_capture/adapters/visual_embedding.py); the
  visual vector is used by the narrative similarity mode, not the clustering view.
- The Streamlit tab renders the dendrogram with Altair rule marks, draws a dashed cut line at
  the selected level, and lays out each cluster as inline HTML thumbnails with `title` tooltips
  so descriptions surface on hover without an extra click. A per-cluster button invokes
  [`ClusterThemeService`](../src/cancel_capture/application/cluster_theme.py) which delegates to
  the structured
  [`OpenAIClusterThemeProvider`](../src/cancel_capture/adapters/openai_provider.py).

## Narrative

- Anchor selection is either explicit (dropdown) or randomized
  ([`NarrativeSelectionService.random_anchor`](../src/cancel_capture/application/narrative_selection.py)),
  with a seeded `random.Random` so a reroll button remains reproducible during a session.
  Companion selection samples uniformly without replacement from every sibling below a
  configurable maximum cosine similarity, using the current `SimilarityMode` (semantic, visual,
  or hybrid). Same-photo siblings are always excluded so the pool never seeds the story with the
  anchor's neighbours.
- The
  [`NarrativeExperimentService`](../src/cancel_capture/application/narrative_experiment.py)
  orchestrates: current-events research via
  [`OpenAICurrentNewsProvider`](../src/cancel_capture/adapters/openai_provider.py) (Responses API
  + built-in `web_search` tool), prompt assembly through
  [`render_narrative_user_prompt`](../src/cancel_capture/prompts/narrative.py), and structured
  narrative generation via
  [`OpenAINarrativeProvider`](../src/cancel_capture/adapters/openai_provider.py). The anchor is
  weighted more heavily than companions by default (2.5 vs. 1.0); the weights and
  reading-minute → target-words mapping are pure functions so alternative strategies can share
  the same orchestration.
- Every generated story is persisted as an atomic, private Markdown artifact under
  `data/narratives/<attempt-id>.md` by
  [`MarkdownNarrativeStore`](../src/cancel_capture/adapters/markdown_narratives.py). Files are
  written with `O_EXCL`, `fsync`, and 0o600 permissions; front matter records prompts, sampling
  parameters, and web citations so historical attempts can be inspected or reused.
- Prompts (vision, archival, cluster theme, narrative system, per-strategy, current-news brief)
  live under [`cancel_capture.prompts`](../src/cancel_capture/prompts/__init__.py) so tuning
  does not require editing adapter code. Test coverage exercises the pure prompt rendering, the
  store round-trip, the selection service, and the structured-output paths of every new
  provider.

## Progress reporting

- Long-running orchestration surfaces through the
  [`ProgressReporter`](../src/cancel_capture/progress.py) `Protocol` (`stage`, `note`,
  `complete`). `NarrativeExperimentService.generate` accepts an optional reporter and wraps each
  remote call with `with_periodic_notes`, which drips fresh wait-lines every few seconds while
  the underlying coroutine is still running. Streamlit binds a `StreamlitProgress` adapter that
  writes into the active `st.status` container; the same protocol lets the bot (or any future
  surface) attach a Telegram-edit reporter without touching application code. The default no-op
  reporter keeps tests and the CLI silent.
- Entertainment lines are pure data in
  [`wait_lines.py`](../src/cancel_capture/wait_lines.py), grouped by stage
  (`news`/`drafting`/`saving`) so tone can be tuned per phase without code changes.

## Provider switching

All external roles (vision, text, embedding, narrative, cluster-theme, current-news) resolve
through the
[`provider_registry`](../src/cancel_capture/provider_registry.py) — a per-role
`dict[str, Callable[[ProviderConfig], T]]` keyed on `ProviderConfig.provider`. Adding a second
backend (e.g. Azure OpenAI, an on-prem endpoint) is a one-line entry per role, and unknown
provider names raise a `ConfigurationError` that lists the registered choices.

## Related pages

- [Implementation overview](implementation.md) — index and runtime shape.
- [Persistence and retrieval](persistence.md) — how sign embeddings and narrative artifacts are
  stored.
