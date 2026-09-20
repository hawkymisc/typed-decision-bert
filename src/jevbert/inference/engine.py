"""Inference engine: bounded admission, one worker thread, deadlines (POC_DESIGN 6.4).

GPU work runs on a single dedicated thread so that it never blocks the event loop, and
admission is bounded so that the queue cannot grow without limit (spec 12.3). A result
belongs to exactly one request: every job has its own future and its own cancel token,
so an abandoned computation can never be handed to somebody else.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

from jevbert.api.errors import DeadlineExceededError, OverloadedError
from jevbert.backends.base import CancelToken, InferenceCancelled

logger = logging.getLogger("jevbert.engine")

T = TypeVar("T")


class InferenceEngine:
    """Serializes backend work onto one thread with bounded admission."""

    def __init__(self, *, max_pending_requests: int = 8, request_deadline_seconds: float = 30.0):
        if max_pending_requests < 1:
            raise ValueError("max_pending_requests must be at least one")
        if request_deadline_seconds <= 0:
            raise ValueError("request_deadline_seconds must be positive")
        self._max_pending = max_pending_requests
        self._deadline_seconds = request_deadline_seconds
        self._executor: ThreadPoolExecutor | None = None
        self._pending = 0
        # Admission is counted on the event loop and released on the worker thread.
        self._lock = threading.Lock()

    @property
    def request_deadline_seconds(self) -> float:
        return self._deadline_seconds

    @property
    def pending(self) -> int:
        return self._pending

    def start(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="jevbert-inference"
            )

    def shutdown(self, *, wait: bool = True) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=wait)

    def deadline_from(self, started_at: float) -> float:
        return started_at + self._deadline_seconds

    async def run(self, work: Callable[[CancelToken], T], *, deadline: float | None = None) -> T:
        """Run ``work`` on the worker thread.

        Raises:
            OverloadedError: admission limit reached (529).
            DeadlineExceededError: the deadline passed before the result arrived (504).
        """
        executor = self._executor
        if executor is None:
            raise RuntimeError("The inference engine is not running")

        cancel = CancelToken()
        timeout = None if deadline is None else deadline - time.monotonic()
        if timeout is not None and timeout <= 0.0:
            raise DeadlineExceededError("The request deadline passed before dispatch.")

        # Counts queued and in-flight requests together (POC_DESIGN 6.4). The slot is
        # held until the worker itself finishes, so a request that already gave up on
        # its deadline does not let more work in than the single thread can serve.
        with self._lock:
            if self._pending >= self._max_pending:
                raise OverloadedError("The inference queue is full; retry shortly.")
            self._pending += 1

        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(executor, self._guard(work), cancel)
        except RuntimeError:
            self._release_slot()
            raise
        future.add_done_callback(self._consume_exception)

        try:
            # Shielded so that abandoning the wait never cancels the shared worker in a
            # way that could surface in another request; cancellation is cooperative.
            return await asyncio.wait_for(asyncio.shield(future), timeout)
        except TimeoutError:
            cancel.cancel()
            raise DeadlineExceededError(
                "The server deadline passed while the request was being processed."
            ) from None
        except asyncio.CancelledError:
            cancel.cancel()
            raise

    def _guard(self, work: Callable[[CancelToken], T]) -> Callable[[CancelToken], T]:
        """Release the admission slot on the worker thread, however the job ends."""

        def runner(cancel: CancelToken) -> T:
            try:
                return work(cancel)
            finally:
                self._release_slot()

        return runner

    def _release_slot(self) -> None:
        with self._lock:
            self._pending -= 1

    def _consume_exception(self, future: Future[object] | asyncio.Future[object]) -> None:
        if future.cancelled():
            return
        error = future.exception()
        if error is None:
            return
        if isinstance(error, InferenceCancelled):
            logger.info("inference job stopped after cancellation: %s", error)
        else:
            # Retrieved here so the loop does not report it as never consumed; the
            # awaiting request receives and reports the same exception.
            logger.debug("inference job finished with %s", type(error).__name__)
