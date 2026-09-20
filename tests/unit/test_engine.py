"""Inference engine: admission, deadline, cancellation (POC_DESIGN 6.4; spec 12.3)."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator

import pytest

from jevbert.api.errors import DeadlineExceededError, OverloadedError
from jevbert.backends.base import CancelToken, InferenceCancelled
from jevbert.inference.engine import InferenceEngine


@pytest.fixture
def engine() -> Iterator[InferenceEngine]:
    instance = InferenceEngine(max_pending_requests=1, request_deadline_seconds=30.0)
    instance.start()
    try:
        yield instance
    finally:
        instance.shutdown(wait=False)


class TestAdmission:
    async def test_work_runs_and_returns_its_own_result(self, engine: InferenceEngine) -> None:
        assert await engine.run(lambda cancel: [1.0, 2.0]) == [1.0, 2.0]

    async def test_queue_full_is_refused_rather_than_queued(
        self, engine: InferenceEngine
    ) -> None:
        started = threading.Event()
        release = threading.Event()

        def slow(cancel: CancelToken) -> list[float]:
            started.set()
            release.wait(5.0)
            return [1.0]

        task = asyncio.create_task(engine.run(slow))
        await asyncio.to_thread(started.wait, 5.0)

        with pytest.raises(OverloadedError):
            await engine.run(lambda cancel: [0.0])

        release.set()
        assert await task == [1.0]

    async def test_capacity_is_released_after_completion(self, engine: InferenceEngine) -> None:
        for _ in range(3):
            assert await engine.run(lambda cancel: [0.0]) == [0.0]
        # The release runs in a done callback, so give the loop one turn to settle.
        await asyncio.sleep(0)
        assert engine.pending == 0

    async def test_capacity_is_released_after_a_failure(self, engine: InferenceEngine) -> None:
        def boom(cancel: CancelToken) -> list[float]:
            raise RuntimeError("backend exploded")

        with pytest.raises(RuntimeError):
            await engine.run(boom)
        await asyncio.sleep(0)
        assert engine.pending == 0
        assert await engine.run(lambda cancel: [0.0]) == [0.0]


class TestDeadline:
    async def test_expired_deadline_is_refused_before_dispatch(
        self, engine: InferenceEngine
    ) -> None:
        dispatched = threading.Event()

        def work(cancel: CancelToken) -> list[float]:
            dispatched.set()
            return [0.0]

        with pytest.raises(DeadlineExceededError):
            await engine.run(work, deadline=time.monotonic() - 1.0)
        assert not dispatched.is_set()

    async def test_slow_work_times_out_and_is_cancelled(self, engine: InferenceEngine) -> None:
        observed = threading.Event()

        def slow(cancel: CancelToken) -> list[float]:
            for _ in range(500):
                if cancel.cancelled:
                    observed.set()
                    raise InferenceCancelled()
                time.sleep(0.01)
            return [0.0]

        with pytest.raises(DeadlineExceededError):
            await engine.run(slow, deadline=time.monotonic() + 0.05)

        # The cancel token must actually reach the worker (spec 12.3).
        assert await asyncio.to_thread(observed.wait, 5.0)

    async def test_deadline_from_start_time(self, engine: InferenceEngine) -> None:
        started = time.monotonic()
        assert engine.deadline_from(started) == started + engine.request_deadline_seconds


class TestResultIsolation:
    async def test_each_request_receives_only_its_own_result(
        self, engine: InferenceEngine
    ) -> None:
        # spec 12.3: an abandoned computation must never be handed to another request.
        wide = InferenceEngine(max_pending_requests=8, request_deadline_seconds=30.0)
        wide.start()
        try:
            results = await asyncio.gather(
                *(wide.run(lambda cancel, n=index: [float(n)]) for index in range(8))
            )
        finally:
            wide.shutdown(wait=False)
        assert results == [[float(index)] for index in range(8)]


class TestLifecycle:
    def test_run_requires_a_started_engine(self) -> None:
        stopped = InferenceEngine()
        with pytest.raises(RuntimeError):
            asyncio.run(stopped.run(lambda cancel: [0.0]))

    def test_invalid_configuration_is_refused(self) -> None:
        with pytest.raises(ValueError):
            InferenceEngine(max_pending_requests=0)
        with pytest.raises(ValueError):
            InferenceEngine(request_deadline_seconds=0.0)
