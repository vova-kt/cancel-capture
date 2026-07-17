from __future__ import annotations

import json
import os
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from cancel_capture.narrative_models import (
    NarrativeArtifact,
    NarrativeArtifactMetadata,
    StoredNarrativeArtifact,
    WebCitation,
    validate_narrative_identifier,
)

_SCHEMA_VERSION = 1
_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
_BODY_MARKER = "<!-- cancel-capture:narrative-body -->"


class NarrativeArtifactFormatError(ValueError):
    pass


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("Narrative creation time must be ISO 8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Narrative creation time must include a timezone")
    return parsed


class MarkdownNarrativeStore:
    def __init__(self, data_dir: Path, *, max_artifact_bytes: int = _MAX_ARTIFACT_BYTES) -> None:
        if max_artifact_bytes <= 0:
            raise ValueError("Maximum narrative artifact size must be positive")
        self._directory = data_dir / "narratives"
        self._max_artifact_bytes = max_artifact_bytes
        self._ensure_private_directory()

    def save(self, artifact: NarrativeArtifact) -> StoredNarrativeArtifact:
        serialized = self._serialize(artifact).encode("utf-8")
        if len(serialized) > self._max_artifact_bytes:
            raise ValueError("Narrative artifact exceeds the configured size limit")

        filename = f"{artifact.metadata.attempt_id}.md"
        target = self._directory / filename
        temporary = self._directory / f".{filename}.{uuid4().hex}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(serialized)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, target)
            os.chmod(target, 0o600)
            self._sync_directory()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

        relative_path = f"narratives/{filename}"
        return self.read(relative_path)

    def read(self, relative_path: str) -> StoredNarrativeArtifact:
        attempt_id = self._attempt_id_from_path(relative_path)
        path = self._directory / f"{attempt_id}.md"
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise NarrativeArtifactFormatError(
                f"Narrative artifact cannot be opened safely: {relative_path}"
            ) from error
        try:
            details = os.fstat(descriptor)
            if not stat.S_ISREG(details.st_mode):
                raise NarrativeArtifactFormatError("Narrative artifact is not a regular file")
            if details.st_size > self._max_artifact_bytes:
                raise NarrativeArtifactFormatError("Narrative artifact exceeds the size limit")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                encoded = stream.read(self._max_artifact_bytes + 1)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if len(encoded) > self._max_artifact_bytes:
            raise NarrativeArtifactFormatError("Narrative artifact exceeds the size limit")
        try:
            contents = encoded.decode("utf-8")
        except UnicodeDecodeError as error:
            raise NarrativeArtifactFormatError("Narrative artifact is not valid UTF-8") from error
        artifact = self._parse(contents)
        if artifact.metadata.attempt_id != attempt_id:
            raise NarrativeArtifactFormatError(
                "Narrative attempt ID does not match its artifact filename"
            )
        return StoredNarrativeArtifact(relative_path=relative_path, artifact=artifact)

    def list_artifacts(self) -> tuple[StoredNarrativeArtifact, ...]:
        records: list[StoredNarrativeArtifact] = []
        for path in self._directory.iterdir():
            match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_-]{0,127})\.md", path.name)
            if match is None or path.is_symlink():
                continue
            records.append(self.read(f"narratives/{path.name}"))
        records.sort(
            key=lambda record: (
                _parse_timestamp(record.artifact.metadata.created_at),
                record.artifact.metadata.attempt_id,
            ),
            reverse=True,
        )
        return tuple(records)

    def _ensure_private_directory(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        details = self._directory.lstat()
        if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
            raise ValueError("Narrative storage path must be a real directory")
        self._directory.chmod(0o700)

    def _sync_directory(self) -> None:
        descriptor = os.open(self._directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _attempt_id_from_path(relative_path: str) -> str:
        candidate = Path(relative_path)
        if candidate.parts != ("narratives", candidate.name) or candidate.suffix != ".md":
            raise ValueError("Narrative artifact path must stay inside the narratives directory")
        attempt_id = candidate.stem
        validate_narrative_identifier(attempt_id, "Narrative attempt ID")
        return attempt_id

    @staticmethod
    def _serialize(artifact: NarrativeArtifact) -> str:
        metadata = artifact.metadata
        header = {
            "schema_version": _SCHEMA_VERSION,
            "title": artifact.title.strip(),
            "description": artifact.description.strip(),
            "metadata": {
                "attempt_id": metadata.attempt_id,
                "created_at": metadata.created_at,
                "anchor_sign_id": metadata.anchor_sign_id,
                "source_sign_ids": list(metadata.source_sign_ids),
                "strategy": metadata.strategy,
                "language": metadata.language,
                "reading_minutes": metadata.reading_minutes,
                "similarity_mode": metadata.similarity_mode,
                "similarity_threshold": metadata.similarity_threshold,
                "semantic_weight": metadata.semantic_weight,
                "random_seed": metadata.random_seed,
                "provider": metadata.provider,
                "model": metadata.model,
                "provider_namespace": metadata.provider_namespace,
                "system_prompt": metadata.system_prompt,
                "user_prompt": metadata.user_prompt,
                "web_citations": [
                    {"title": citation.title, "url": citation.url}
                    for citation in metadata.web_citations
                ],
                "output_version": metadata.output_version,
            },
        }
        encoded_header = json.dumps(header, ensure_ascii=False, indent=2, sort_keys=True)
        title = artifact.title.strip()
        description = artifact.description.strip()
        body = artifact.body_markdown.strip()
        return (
            f"---\n{encoded_header}\n---\n# {title}\n\n{description}\n\n{_BODY_MARKER}\n\n{body}\n"
        )

    @classmethod
    def _parse(cls, contents: str) -> NarrativeArtifact:
        if not contents.startswith("---\n"):
            raise NarrativeArtifactFormatError("Narrative artifact has no JSON front matter")
        header_text, separator, rendered = contents[4:].partition("\n---\n")
        if not separator:
            raise NarrativeArtifactFormatError("Narrative artifact front matter is incomplete")
        try:
            decoded = cast(object, json.loads(header_text))
        except json.JSONDecodeError as error:
            raise NarrativeArtifactFormatError(
                "Narrative artifact front matter is invalid JSON"
            ) from error
        header = cls._string_dict(decoded, "Narrative artifact front matter")
        expected_header = {"schema_version", "title", "description", "metadata"}
        if set(header) != expected_header:
            raise NarrativeArtifactFormatError("Narrative artifact front matter fields are invalid")
        schema_version = cls._integer(header["schema_version"], "schema_version")
        if schema_version != _SCHEMA_VERSION:
            raise NarrativeArtifactFormatError(
                f"Unsupported narrative artifact schema version: {schema_version}"
            )
        title = cls._string(header["title"], "title")
        description = cls._string(header["description"], "description")
        raw_metadata = cls._string_dict(header["metadata"], "metadata")
        expected_metadata = {
            "attempt_id",
            "created_at",
            "anchor_sign_id",
            "source_sign_ids",
            "strategy",
            "language",
            "reading_minutes",
            "similarity_mode",
            "similarity_threshold",
            "semantic_weight",
            "random_seed",
            "provider",
            "model",
            "provider_namespace",
            "system_prompt",
            "user_prompt",
            "web_citations",
            "output_version",
        }
        if set(raw_metadata) != expected_metadata:
            raise NarrativeArtifactFormatError("Narrative artifact metadata fields are invalid")
        random_seed_value = raw_metadata["random_seed"]
        random_seed = (
            None if random_seed_value is None else cls._integer(random_seed_value, "random_seed")
        )
        try:
            metadata = NarrativeArtifactMetadata(
                attempt_id=cls._string(raw_metadata["attempt_id"], "attempt_id"),
                created_at=cls._string(raw_metadata["created_at"], "created_at"),
                anchor_sign_id=cls._string(raw_metadata["anchor_sign_id"], "anchor_sign_id"),
                source_sign_ids=cls._string_tuple(
                    raw_metadata["source_sign_ids"], "source_sign_ids"
                ),
                strategy=cls._string(raw_metadata["strategy"], "strategy"),
                language=cls._string(raw_metadata["language"], "language"),
                reading_minutes=cls._integer(raw_metadata["reading_minutes"], "reading_minutes"),
                similarity_mode=cls._string(raw_metadata["similarity_mode"], "similarity_mode"),
                similarity_threshold=cls._number(
                    raw_metadata["similarity_threshold"], "similarity_threshold"
                ),
                semantic_weight=cls._number(raw_metadata["semantic_weight"], "semantic_weight"),
                random_seed=random_seed,
                provider=cls._string(raw_metadata["provider"], "provider"),
                model=cls._string(raw_metadata["model"], "model"),
                provider_namespace=cls._string(
                    raw_metadata["provider_namespace"], "provider_namespace"
                ),
                system_prompt=cls._string(raw_metadata["system_prompt"], "system_prompt"),
                user_prompt=cls._string(raw_metadata["user_prompt"], "user_prompt"),
                web_citations=cls._citations(raw_metadata["web_citations"]),
                output_version=cls._integer(raw_metadata["output_version"], "output_version"),
            )
        except ValueError as error:
            raise NarrativeArtifactFormatError(str(error)) from error
        rendered_prefix = f"# {title}\n\n{description}\n\n{_BODY_MARKER}\n\n"
        if not rendered.startswith(rendered_prefix):
            raise NarrativeArtifactFormatError(
                "Narrative title or description does not match its front matter"
            )
        body = rendered[len(rendered_prefix) :].rstrip("\n")
        try:
            return NarrativeArtifact(
                title=title,
                description=description,
                body_markdown=body,
                metadata=metadata,
            )
        except ValueError as error:
            raise NarrativeArtifactFormatError(str(error)) from error

    @staticmethod
    def _string_dict(value: object, label: str) -> dict[str, object]:
        if not isinstance(value, dict):
            raise NarrativeArtifactFormatError(f"{label} must be a JSON object")
        raw = cast(dict[object, object], value)
        if not all(isinstance(key, str) for key in raw):
            raise NarrativeArtifactFormatError(f"{label} must be a JSON object")
        return cast(dict[str, object], value)

    @staticmethod
    def _string(value: object, label: str) -> str:
        if not isinstance(value, str):
            raise NarrativeArtifactFormatError(f"{label} must be a string")
        return value

    @staticmethod
    def _string_tuple(value: object, label: str) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise NarrativeArtifactFormatError(f"{label} must be a string list")
        raw = cast(list[object], value)
        if not all(isinstance(item, str) for item in raw):
            raise NarrativeArtifactFormatError(f"{label} must be a string list")
        return tuple(cast(list[str], value))

    @staticmethod
    def _integer(value: object, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise NarrativeArtifactFormatError(f"{label} must be an integer")
        return value

    @staticmethod
    def _number(value: object, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NarrativeArtifactFormatError(f"{label} must be a number")
        return float(value)

    @classmethod
    def _citations(cls, value: object) -> tuple[WebCitation, ...]:
        if not isinstance(value, list):
            raise NarrativeArtifactFormatError("web_citations must be a list")
        citations: list[WebCitation] = []
        for entry in cast(list[object], value):
            payload = cls._string_dict(entry, "web citation")
            if set(payload) != {"title", "url"}:
                raise NarrativeArtifactFormatError("Web citation fields are invalid")
            try:
                citations.append(
                    WebCitation(
                        title=cls._string(payload["title"], "citation title"),
                        url=cls._string(payload["url"], "citation URL"),
                    )
                )
            except ValueError as error:
                raise NarrativeArtifactFormatError(str(error)) from error
        return tuple(citations)
