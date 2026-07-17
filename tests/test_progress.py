from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from cancel_capture.progress import NullProgress, ProgressReporter, with_periodic_notes


@dataclass
class RecordingReporter:
    stages: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    completions: list[tuple[str, bool]] = field(default_factory=list)

    def stage(self, key: str, label: str) -> None:
        self.stages.append((key, label))

    def note(self, text: str) -> None:
        self.notes.append(text)

    def complete(self, label: str, *, ok: bool = True) -> None:
        self.completions.append((label, ok))


def test_null_progress_conforms_to_protocol_and_stays_silent() -> None:
    reporter: ProgressReporter = NullProgress()
    reporter.stage("news", "reading")
    reporter.note("chatter")
    reporter.complete("done")


async def test_periodic_notes_emit_while_task_runs_and_stop_on_completion() -> None:
    reporter = RecordingReporter()
    counter = {"value": 0}

    def note_provider() -> str:
        counter["value"] += 1
        return f"ping {counter['value']}"

    async def slow_task() -> str:
        await asyncio.sleep(0.35)
        return "done"

    result = await with_periodic_notes(
        slow_task(),
        note_provider,
        interval_seconds=0.1,
        reporter=reporter,
    )

    assert result == "done"
    assert reporter.notes  # at least one ping fired
    assert all(text.startswith("ping ") for text in reporter.notes)


async def test_periodic_notes_stay_silent_for_immediately_completed_work() -> None:
    reporter = RecordingReporter()

    async def instant() -> int:
        return 7

    result = await with_periodic_notes(
        instant(),
        lambda: "should not appear",
        interval_seconds=0.5,
        reporter=reporter,
    )

    assert result == 7
    assert reporter.notes == []


async def test_periodic_notes_bypass_scheduler_when_interval_disabled() -> None:
    reporter = RecordingReporter()

    async def instant() -> str:
        return "value"

    result = await with_periodic_notes(
        instant(),
        lambda: "silent",
        interval_seconds=0.0,
        reporter=reporter,
    )

    assert result == "value"
    assert reporter.notes == []


async def test_periodic_notes_cancel_the_task_when_the_outer_scope_raises() -> None:
    reporter = RecordingReporter()

    async def endless() -> None:
        await asyncio.sleep(30)

    coro = endless()
    with pytest.raises(RuntimeError, match="stop"):

        async def raiser() -> None:
            async def failing_provider() -> str:  # pyright: ignore [reportUnusedFunction]
                return "unused"

            def note_provider() -> str:
                raise RuntimeError("stop")

            await with_periodic_notes(
                coro,
                note_provider,
                interval_seconds=0.05,
                reporter=reporter,
            )

        await raiser()
