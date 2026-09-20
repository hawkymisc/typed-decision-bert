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
        # The slot is released in the worker's own `finally`, before the future
        # resolves (D5), so it is free the moment the await returns. Nothing has to
        # give the loop an extra turn.
        for _ in range(3):
            assert await engine.run(lambda cancel: [0.0]) == [0.0]
            assert engine.pending == 0

    async def test_capacity_is_released_after_a_failure(self, engine: InferenceEngine) -> None:
        def boom(cancel: CancelToken) -> list[float]:
            raise RuntimeError("backend exploded")

        with pytest.raises(RuntimeError):
            await engine.run(boom)
        assert engine.pending == 0
        assert await engine.run(lambda cancel: [0.0]) == [0.0]

    async def test_encoding_shares_the_same_admission_budget(
        self, engine: InferenceEngine
    ) -> None:
        # S-H2: tokenization must not be able to pile up in front of a busy worker.
        started = threading.Event()
        release = threading.Event()

        def slow(cancel: CancelToken) -> str:
            started.set()
            release.wait(5.0)
            return "encoded"

        task = asyncio.create_task(engine.run_encoding(slow))
        await asyncio.to_thread(started.wait, 5.0)
        with pytest.raises(OverloadedError):
            await engine.run(lambda cancel: [0.0])

        release.set()
        assert await task == "encoded"
        assert engine.pending == 0


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


class TestSeparateThreads:
    async def test_encoding_and_inference_use_different_single_threads(
        self, engine: InferenceEngine
    ) -> None:
        def name(cancel: CancelToken) -> str:
            return threading.current_thread().name

        encoder = await engine.run_encoding(name)
        worker = await engine.run(name)
        assert encoder.startswith("jevbert-encoder")
        assert worker.startswith("jevbert-inference")
        assert encoder != worker

    async def test_each_pool_keeps_one_thread(self, engine: InferenceEngine) -> None:
        # A tokenizer and a device are both single-owner resources.
        wide = InferenceEngine(max_pending_requests=8, request_deadline_seconds=30.0)
        wide.start()
        try:

            def name(cancel: CancelToken) -> str:
                time.sleep(0.01)
                return threading.current_thread().name

            encoders = await asyncio.gather(*(wide.run_encoding(name) for _ in range(4)))
            workers = await asyncio.gather(*(wide.run(name) for _ in range(4)))
        finally:
            wide.shutdown(wait=False)
        assert len(set(encoders)) == 1
        assert len(set(workers)) == 1


class TestBlockingRunners:
    """Used by bundle warmup, which runs on the loading thread (A-F10)."""

    def test_blocking_work_runs_on_the_inference_worker(
        self, engine: InferenceEngine
    ) -> None:
        name = engine.run_blocking(lambda cancel: threading.current_thread().name)
        assert name.startswith("jevbert-inference")

    def test_blocking_encoding_runs_on_the_encoder(self, engine: InferenceEngine) -> None:
        name = engine.run_encoding_blocking(lambda cancel: threading.current_thread().name)
        assert name.startswith("jevbert-encoder")

    def test_warmup_does_not_consume_an_admission_slot(
        self, engine: InferenceEngine
    ) -> None:
        # max_pending_requests is 1 in this fixture: if warmup took the slot, a
        # request arriving during startup would see a 529 for no reason.
        engine.run_blocking(lambda cancel: None)
        assert engine.pending == 0

    def test_blocking_work_on_a_stopped_engine_is_refused(self) -> None:
        stopped = InferenceEngine()
        with pytest.raises(RuntimeError):
            stopped.run_blocking(lambda cancel: None)
        with pytest.raises(RuntimeError):
            stopped.run_encoding_blocking(lambda cancel: None)


class TestLifecycle:
    def test_run_requires_a_started_engine(self) -> None:
        stopped = InferenceEngine()
        with pytest.raises(RuntimeError):
            asyncio.run(stopped.run(lambda cancel: [0.0]))

    def test_encoding_requires_a_started_engine(self) -> None:
        stopped = InferenceEngine()
        with pytest.raises(RuntimeError):
            asyncio.run(stopped.run_encoding(lambda cancel: [0.0]))

    def test_shutdown_leaves_no_admission_owed(self) -> None:
        # A-F13: a job still running after wait=False releases its slot into a stopped
        # engine. The counter must not go negative and leave a restarted engine
        # refusing work it could serve.
        engine = InferenceEngine(max_pending_requests=1, request_deadline_seconds=30.0)
        engine.start()
        engine.shutdown(wait=False)
        assert engine.pending == 0

        engine.start()
        assert asyncio.run(engine.run(lambda cancel: [0.0])) == [0.0]
        assert engine.pending == 0
        engine.shutdown(wait=True)

    def test_start_is_idempotent(self) -> None:
        engine = InferenceEngine()
        engine.start()
        try:
            first = asyncio.run(engine.run(lambda cancel: threading.current_thread().name))
            engine.start()
            second = asyncio.run(engine.run(lambda cancel: threading.current_thread().name))
            assert first == second
        finally:
            engine.shutdown(wait=False)

    def test_invalid_configuration_is_refused(self) -> None:
        with pytest.raises(ValueError):
            InferenceEngine(max_pending_requests=0)
        with pytest.raises(ValueError):
            InferenceEngine(request_deadline_seconds=0.0)
