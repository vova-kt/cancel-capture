from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol


class ProgressReporter(Protocol):
    """Boundary for surfacing long-running work to a UI.

    Implementations must be safe to call from an asyncio event loop; the reporter itself is not
    async because Streamlit widgets and Telegram edits are both invoked as synchronous side
    effects.
    """

    def stage(self, key: str, label: str) -> None: ...

    def note(self, text: str) -> None: ...

    def complete(self, label: str, *, ok: bool = True) -> None: ...


class NullProgress:
    def stage(self, key: str, label: str) -> None:
        return None

    def note(self, text: str) -> None:
        return None

    def complete(self, label: str, *, ok: bool = True) -> None:
        return None


async def with_periodic_notes[T](
    coroutine: Awaitable[T],
    note_provider: Callable[[], str],
    *,
    interval_seconds: float,
    reporter: ProgressReporter,
) -> T:
    """Await ``coroutine`` while emitting fresh ``note_provider()`` lines every interval.

    Notes only fire when the underlying work is still running past ``interval_seconds`` — quick
    calls stay silent so tests and instant paths don't spam the UI. A non-positive interval
    disables pings entirely.
    """
    if interval_seconds <= 0:
        return await coroutine

    async def _wrap() -> T:
        return await coroutine

    task: asyncio.Task[T] = asyncio.create_task(_wrap())
    try:
        while True:
            done, _pending = await asyncio.wait({task}, timeout=interval_seconds)
            if task in done:
                return task.result()
            reporter.note(note_provider())
    except BaseException:
        task.cancel()
        raise
