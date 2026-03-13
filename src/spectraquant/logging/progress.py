"""Progress helpers for quiet/verbose CLI workflows."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from rich.progress import track

T = TypeVar("T")


def progress_iter(items: Iterable[T], description: str, enabled: bool) -> Iterator[T]:
    """Iterate *items* with a Rich progress bar when enabled.

    In verbose mode we prefer raw logs, so this falls back to the original
    iterable without wrapping.
    """

    if not enabled:
        yield from items
        return
    yield from track(items, description=description)

