"""Tests reid_worker.py — priorité et fraîcheur des imagettes soumises.

Le symptôme visible quand cette file se comporte mal : des bovins bien
détectés qui restent affichés "?" parce que leur embedding n'arrive jamais.
"""
import queue
import time

import numpy as np
import pytest

from reid_worker import ReIDWorker


class _FakeReID:
    """Moteur factice : embedding déterministe, latence réglable."""
    TOTAL_DIM = 4

    def __init__(self, latency: float = 0.0):
        self.latency = latency
        self.seen: list[list[int]] = []

    def get_embedding_batch(self, crops):
        if self.latency:
            time.sleep(self.latency)
        self.seen.append([int(c[0, 0]) for c in crops])
        return [np.ones(self.TOTAL_DIM, dtype=np.float32) for _ in crops]


def _crop(marker: int) -> np.ndarray:
    return np.full((4, 4), marker, dtype=np.uint8)


@pytest.fixture
def worker():
    w = ReIDWorker(_FakeReID())
    yield w
    w.stop()


def test_empty_batch_is_a_noop(worker):
    assert worker.submit_batch({}) is True


def test_embeddings_come_back_keyed_by_track_id(worker):
    worker.submit_batch({7: _crop(7), 9: _crop(9)})
    deadline = time.time() + 2.0
    ready = {}
    while time.time() < deadline and len(ready) < 2:
        ready.update(worker.collect_ready())
        time.sleep(0.01)
    assert set(ready) == {7, 9}


def test_batch_is_capped():
    engine = _FakeReID()
    w = ReIDWorker(engine, max_batch=3)
    try:
        w.submit_batch({i: _crop(i) for i in range(10)})
        deadline = time.time() + 2.0
        while time.time() < deadline and not engine.seen:
            time.sleep(0.01)
        assert engine.seen and len(engine.seen[0]) == 3
    finally:
        w.stop()


def test_priority_ids_are_served_first():
    engine = _FakeReID()
    w = ReIDWorker(engine, max_batch=2)
    try:
        w.submit_batch({1: _crop(1), 2: _crop(2), 3: _crop(3)},
                       priority_ids={3})
        deadline = time.time() + 2.0
        while time.time() < deadline and not engine.seen:
            time.sleep(0.01)
        assert engine.seen and 3 in engine.seen[0]
    finally:
        w.stop()


def test_full_queue_replaces_the_stalest_batch():
    """Régression : la file pleine rejetait les NOUVEAUX crops et le worker
    continuait à traiter des imagettes périmées."""
    engine = _FakeReID(latency=5.0)   # worker occupé, la file se remplit
    w = ReIDWorker(engine, max_queue=1, max_batch=4)
    try:
        for i in range(6):
            assert w.submit_batch({i: _crop(i)}) is True
        # Un seul batch en attente, et c'est le plus récent.
        pending = []
        while True:
            try:
                pending.append(w._queue.get_nowait())
            except queue.Empty:
                break
        assert len(pending) <= 1
        if pending:
            _, items = pending[0]
            assert items[0][0] == 5      # le dernier soumis
    finally:
        w.stop()


def test_failed_worker_refuses_submissions(worker):
    worker._failed = True
    assert worker.submit_batch({1: _crop(1)}) is False
    assert worker.has_failed() is True
