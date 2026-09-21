"""Inference engine: bounded admission, dedicated threads, deadlines (POC_DESIGN 6.4).

Two single-thread pools sit behind the event loop:

* the **encoder**, where compilation and tokenization run. One thread, because a
  tokenizer is shared mutable state and the PoC does not want to reason about making
  one safe for concurrent use (S-H2).
* the **inference** worker, where the backend scores. One thread, because that is what
  serializes GPU work (spec 12.3).

Both are bounded by the same admission counter, so tokenization cannot pile up ahead of
a busy GPU, and both are covered by the request deadline. A result belongs to exactly
one request: every job has its own future and its own cancel token, so an abandoned
computation can never be handed to somebody else.
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

#: How long a warmup job may block the loading thread before it is given up on. Warmup
#: shares the inference worker with served requests, so it cannot wait forever.
WARMUP_TIMEOUT_SECONDS = 120.0


class InferenceEngine:
    """Serializes encoding and backend work onto dedicated threads."""

    def __init__(self, *, max_pending_requests: int = 8, request_deadline_seconds: float = 30.0):
        if max_pending_requests < 1:
            raise ValueError("max_pending_requests must be at least one")
        if request_deadline_seconds <= 0:
            raise ValueError("request_deadline_seconds must be positive")
        self._max_pending = max_pending_requests
        self._deadline_seconds = request_deadline_seconds
        self._executor: ThreadPoolExecutor | None = None
        self._encoder: ThreadPoolExecutor | None = None
        self._pending = 0
        # Admission is taken on the event loop and released on the worker thread.
        self._lock = threading.Lock()

    @property
    def request_deadline_seconds(self) -> float:
        return self._deadline_seconds

    @property
    def pending(self) -> int:
        """Requests currently admitted, whether queued or running."""
        with self._lock:
            return self._pending

    def start(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="jevbert-inference"
            )
        if self._encoder is None:
            self._encoder = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="jevbert-encoder"
            )

    def shutdown(self, *, wait: bool = True) -> None:
        executor, self._executor = self._executor, None
        encoder, self._encoder = self._encoder, None
        for pool in (encoder, executor):
            if pool is not None:
                pool.shutdown(wait=wait)
        with self._lock:
            # A job still running after ``wait=False`` will release its slot into a
            # stopped engine; the counter is reset here and clamped there so the two
            # cannot drive it negative and make a restarted engine refuse work.
            self._pending = 0

    def deadline_from(self, started_at: float) -> float:
        return started_at + self._deadline_seconds

    async def run(self, work: Callable[[CancelToken], T], *, deadline: float | None = None) -> T:
        """Run ``work`` on the inference worker."""
        return await self._submit(self._executor, "inference", work, deadline)

    async def run_encoding(
        self, work: Callable[[CancelToken], T], *, deadline: float | None = None
    ) -> T:
        """Run ``work`` on the encoder thread, under the same admission and deadline."""
        return await self._submit(self._encoder, "encoder", work, deadline)

    def run_blocking(self, work: Callable[[CancelToken], T]) -> T:
        """Run ``work`` on the inference worker from a plain thread.

        Used by bundle warmup, which happens on the loading thread but must still go
        through the one worker that owns the device: a warmup forward pass running
        beside a served request is exactly the concurrent GPU access the single worker
        exists to prevent (A-F10). Warmup is not a request, so it does not take an
        admission slot; it is serialized by the worker itself.
        """
        executor = self._executor
        if executor is None:
            raise RuntimeError("The inference engine is not running")
        return executor.submit(work, CancelToken()).result(WARMUP_TIMEOUT_SECONDS)

    def run_encoding_blocking(self, work: Callable[[CancelToken], T]) -> T:
        """The encoder-thread counterpart of :meth:`run_blocking`."""
        encoder = self._encoder
        if encoder is None:
            raise RuntimeError("The inference engine is not running")
        return encoder.submit(work, CancelToken()).result(WARMUP_TIMEOUT_SECONDS)

    async def _submit(
        self,
        executor: ThreadPoolExecutor | None,
        stage: str,
        work: Callable[[CancelToken], T],
        deadline: float | None,
    ) -> T:
        """Admit, dispatch and await one stage of a request.

        Raises:
            OverloadedError: admission limit reached (529).
            DeadlineExceededError: the deadline passed before the result arrived (504).
        """
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
        future.add_done_callback(lambda done: self._consume_exception(stage, done))

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
            self._pending = max(0, self._pending - 1)

    def _consume_exception(
        self, stage: str, future: Future[object] | asyncio.Future[object]
    ) -> None:
        if future.cancelled():
            return
        error = future.exception()
        if error is None:
            return
        if isinstance(error, InferenceCancelled):
            logger.info("%s job stopped after cancellation", stage)
        else:
            # Retrieved here so the loop does not report it as never consumed; the
            # awaiting request receives and reports the same exception. The message is
            # not logged: a tokenizer or torch error can quote the input (spec 15.2).
            logger.debug("%s job finished with %s", stage, type(error).__name__)
