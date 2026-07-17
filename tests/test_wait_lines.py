from __future__ import annotations

from random import Random

import pytest

from cancel_capture.wait_lines import (
    NARRATIVE_STAGE_LINES,
    random_wait_line,
    wait_lines_for,
)


def test_every_stage_has_short_non_empty_lines() -> None:
    for stage, lines in NARRATIVE_STAGE_LINES.items():
        assert lines, f"stage {stage!r} is empty"
        for line in lines:
            assert line.strip(), f"stage {stage!r} contains blank line"
            assert len(line) <= 120, f"stage {stage!r} line too long: {line!r}"


def test_wait_lines_for_raises_for_unknown_stage() -> None:
    with pytest.raises(KeyError):
        wait_lines_for("no-such-stage")


def test_random_wait_line_is_deterministic_for_a_seed() -> None:
    first = random_wait_line("drafting", rng=Random(11))
    repeated = random_wait_line("drafting", rng=Random(11))
    other = random_wait_line("drafting", rng=Random(12))

    assert first == repeated
    assert first in NARRATIVE_STAGE_LINES["drafting"]
    assert other in NARRATIVE_STAGE_LINES["drafting"]


def test_random_wait_line_eventually_visits_every_option() -> None:
    stage = "news"
    rng = Random(0)
    seen: set[str] = set()
    for _ in range(1000):
        seen.add(random_wait_line(stage, rng=rng))
    assert seen == set(NARRATIVE_STAGE_LINES[stage])
