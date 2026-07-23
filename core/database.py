"""
core/database.py
----------------
Stores and persists embeddings of known cattle.
Format: pickle with {name: {embedding: np.array, count: int, first_seen: str}}

Optimizations:
- Cache matrix (N, D) + norms vector, rebuilt on each mutation
- Vectorized match(): 1 BLAS call instead of Python loop × N dot products
"""
import pickle
import os
import numpy as np
from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, '..')
from utils.console import info, ok, warn


class EmbeddingDatabase:
    """Cattle embedding database with vectorized matching."""

    def __init__(self, path: str = "cattle_db.pkl", reid_engine=None):
        """
        Args:
            path: Path to pickle file
            reid_engine: CattleReID instance for component-wise similarity.
                        If None, falls back to classical cosine.
        """
        self.path = path
        self.animals: dict = {}
        self.reid_engine = reid_engine
        # Vectorized cache for fast match()
        self._matrix: np.ndarray | None = None   # (N, D) float32
        self._names: list[str] = []              # length N, aligned with _matrix
        self._norms: np.ndarray | None = None     # (N,) float32, L2 norms per row
        self._dirty = True                        # rebuild needed
        self.load()

    def _rebuild_cache(self) -> None:
        """Rebuild _matrix/_norms/_names from self.animals. O(N)."""
        if not self.animals:
            self._matrix = None
            self._norms = None
            self._names = []
            self._dirty = False
            return
        names = list(self.animals.keys())
        vecs = [np.asarray(self.animals[n]["embedding"], dtype=np.float32).ravel()
                for n in names]
        # Check consistent dimensions
        dims = {v.size for v in vecs}
        if len(dims) > 1:
            warn(f"[DB] Inconsistent dimensions in cache: {dims}, purging.")
            self.animals = {n: d for n, d in self.animals.items()
                            if np.asarray(d["embedding"]).ravel().size == max(dims)}
            names = list(self.animals.keys())
            vecs = [np.asarray(self.animals[n]["embedding"], dtype=np.float32).ravel()
                    for n in names]
        M = np.stack(vecs).astype(np.float32, copy=False)
        norms = np.linalg.norm(M, axis=1).astype(np.float32)
        self._matrix = M
        self._norms = norms
        self._names = names
        self._dirty = False

    def _ensure_cache(self) -> None:
        if self._dirty or self._matrix is None:
            self._rebuild_cache()

    def load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self.animals = pickle.load(f)
                ok(f"[DB] {len(self.animals)} animals loaded from {self.path}")
            except Exception as e:
                warn(f"[DB] Read error ({e}), new database.")
                self.animals = {}
        else:
            info(f"[DB] New database (file {self.path} will be created)")
        self._dirty = True

    def save(self) -> None:
        """Save database to pickle file."""
        try:
            with open(self.path, "wb") as f:
                pickle.dump(self.animals, f)
            ok(f"[DB] Saved {len(self.animals)} animals to {self.path}")
        except Exception as e:
            warn(f"[DB] Save failed: {e}")

    def validate_dim(self, expected_dim: int) -> int:
        """Verify all embeddings have dimension `expected_dim`. Purge if not."""
        if not self.animals:
            return 0
        bad = []
        for name, data in self.animals.items():
            emb = data.get("embedding")
            if emb is None or emb.shape[0] != expected_dim:
                bad.append(name)
        if bad:
            warn(f"[DB] {len(bad)} animals with incompatible embedding (dim != {expected_dim}), purging.")
            for name in bad:
                del self.animals[name]
            self.save()
        self._dirty = True
        return len(self.animals)

    def add(self, name: str, embedding: np.ndarray, first_seen: str | None = None) -> None:
        """Add or update an animal's embedding."""
        if first_seen is None:
            first_seen = datetime.now().isoformat()
        if name in self.animals:
            self.animals[name]["count"] += 1
            # EMA update
            old_emb = self.animals[name]["embedding"]
            alpha = 0.5
            new_emb = alpha * embedding + (1 - alpha) * old_emb
            new_norm = np.linalg.norm(new_emb)
            if new_norm > 0:
                new_emb = new_emb / new_norm
            self.animals[name]["embedding"] = new_emb
        else:
            emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            self.animals[name] = {
                "embedding": emb_norm.astype(np.float32),
                "count": 1,
                "first_seen": first_seen,
            }
        self._dirty = True

    def match(self, embedding: np.ndarray, threshold: float = 0.70) -> tuple[str | None, float]:
        """Find best matching animal. Returns (name, similarity) or (None, 0)."""
        self._ensure_cache()
        if self._matrix is None or self._matrix.shape[0] == 0:
            return None, 0.0
        emb = embedding.astype(np.float32).ravel()
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        # Cosine similarity via dot product
        sims = self._matrix @ emb_norm
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= threshold:
            return self._names[best_idx], best_sim
        return None, best_sim

    def reset(self) -> None:
        """Reset database."""
        self.animals = {}
        self._dirty = True
        if os.path.exists(self.path):
            os.remove(self.path)
        info("[DB] Database reset")

    def __len__(self) -> int:
        return len(self.animals)

    def __repr__(self) -> str:
        return f"EmbeddingDatabase(path={self.path}, animals={len(self.animals)})"
