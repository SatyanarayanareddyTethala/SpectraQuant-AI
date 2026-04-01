"""Asynchronous worker abstraction for control-plane run execution."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class RunTask:
    run_id: str


class Worker(Protocol):
    def submit(self, task: RunTask) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


class InMemoryWorker:
    """Single-process background worker.

    Contract-compatible with future distributed queue worker implementations.
    """

    def __init__(self, handler: Callable[[RunTask], None]) -> None:
        self._handler = handler
        self._queue: queue.Queue[RunTask] = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="sqv3-control-plane-worker")
        self._started = False

    def submit(self, task: RunTask) -> None:
        self._queue.put(task)

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        self._queue.put(RunTask(run_id="__shutdown__"))
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            task = self._queue.get()
            if task.run_id == "__shutdown__":
                self._queue.task_done()
                continue
            try:
                self._handler(task)
            finally:
                self._queue.task_done()
