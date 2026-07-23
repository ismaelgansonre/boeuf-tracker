"""
utils/reid_worker.py
--------------------
Async worker thread for Re-ID embedding computation.
"""
import queue
import threading
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.reid import CattleReID

import sys
sys.path.insert(0, '..')
from utils.console import warn


class ReIDWorker:
    """Async worker thread for computing Re-ID embeddings.

    Usage:
        worker = ReIDWorker(reid_engine)
        worker.submit_batch({det_idx: crop, ...})   # non-blocking
        emb_by_idx = worker.collect_ready()          # non-blocking, may be empty
    """

    def __init__(self, reid_engine: "CattleReID", max_queue: int = 64):
        self.reid = reid_engine
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._results: dict[int, np.ndarray] = {}
        self._stop = threading.Event()
        self._failed = False
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="reid-worker"
        )
        self._thread.start()

    def _loop(self):
        """Consumer: takes batches from queue, computes, stores results."""
        try:
            while not self._stop.is_set():
                try:
                    token, batch = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if batch is None:
                    break  # stop signal
                try:
                    crops = [c for _, c in batch]
                    embeddings = self.reid.get_embedding_batch(crops)
                    with self._lock:
                        for (det_idx, _), emb in zip(batch, embeddings):
                            if emb is not None and emb.shape[0] == self.reid.TOTAL_DIM:
                                self._results[det_idx] = emb
                except Exception as e:
                    warn(f"[ReIDWorker] batch error: {e}")
        except Exception as e:
            warn(f"[ReIDWorker] thread crashed, sync fallback: {e}")
            self._failed = True

    def submit_batch(self, batch: dict) -> None:
        """Submit a batch of crops for processing. Non-blocking."""
        if self._failed:
            return
        items = list(batch.items())
        try:
            self._queue.put_nowait((None, items))
        except queue.Full:
            warn("[ReIDWorker] queue full, dropping batch")

    def collect_ready(self) -> dict[int, np.ndarray]:
        """Collect ready embeddings. Non-blocking, may be empty."""
        if self._failed:
            return {}
        with self._lock:
            results = dict(self._results)
            self._results.clear()
        return results

    def stop(self) -> None:
        """Stop the worker thread."""
        self._stop.set()
        try:
            self._queue.put_nowait((None, None))
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def __del__(self):
        self.stop()

    def __repr__(self) -> str:
        return f"ReIDWorker(failed={self._failed}, queue_size={self._queue.qsize()})"
