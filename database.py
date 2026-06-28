"""
database.py
-----------
Stocke et persiste les embeddings des bovins connus.
Format: pickle avec {nom: {embedding: np.array, count: int, first_seen: str}}

Optimisations:
- Cache d'une matrice (N, D) + vecteur de normes, reconstruit à chaque mutation.
- match() vectorisé: 1 seul BLAS au lieu d'une boucle Python × N dot products.
"""
import pickle
import os
import numpy as np
from datetime import datetime

from console import info, ok, warn


class EmbeddingDatabase:
    """Base simple de vecteurs d'embeddings par bovin."""

    def __init__(self, path: str = "cattle_db.pkl", reid_engine=None):
        """
        reid_engine: instance de CattleReID pour calculer les similarités
        par composante. Si None, fallback sur cosine classique.
        """
        self.path = path
        self.animals: dict = {}
        self.reid_engine = reid_engine
        # Cache vectorisé pour match() rapide
        self._matrix: np.ndarray | None = None   # (N, D) float32
        self._names: list[str] = []               # length N, aligné avec _matrix
        self._norms: np.ndarray | None = None     # (N,) float32, normes L2 par ligne
        self._dirty = True                        # rebuild nécessaire
        self.load()

    def _rebuild_cache(self) -> None:
        """Reconstruit _matrix/_norms/_names à partir de self.animals. O(N)."""
        if not self.animals:
            self._matrix = None
            self._norms = None
            self._names = []
            self._dirty = False
            return
        names = list(self.animals.keys())
        vecs = [np.asarray(self.animals[n]["embedding"], dtype=np.float32).ravel()
                for n in names]
        # Vérif dim cohérente
        dims = {v.size for v in vecs}
        if len(dims) > 1:
            warn(f"[DB] Dimensions incohérentes dans le cache: {dims}, purge.")
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
                ok(f"[DB] {len(self.animals)} animaux charges depuis {self.path}")
            except Exception as e:
                warn(f"[DB] Erreur de lecture ({e}), nouvelle base.")
                self.animals = {}
        else:
            info(f"[DB] Nouvelle base (fichier {self.path} sera cree)")
        self._dirty = True

    def validate_dim(self, expected_dim: int) -> int:
        """
        Vérifie que tous les embeddings ont la dimension `expected_dim`.
        Si non, purge la base (changement de modèle Re-ID).
        Retourne le nombre d'animaux restants.
        """
        if not self.animals:
            return 0
        bad = []
        for name, data in self.animals.items():
            emb = data.get("embedding")
            if emb is None or emb.shape[0] != expected_dim:
                bad.append(name)
        if bad:
            warn(f"[DB] {len(bad)} animaux avec embedding incompatible (dim != {expected_dim}), purge.")
            for name in bad:
                del self.animals[name]
            self.save()
        self._dirty = True
        return len(self.animals)

    def save(self) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(self.animals, f)
        ok(f"[DB] {len(self.animals)} animaux sauvegardes dans {self.path}")

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _similarity(self, embedding: np.ndarray, stored: np.ndarray) -> float:
        """Utilise ReID.compare si dispo, sinon cosine simple."""
        if self.reid_engine is not None:
            return self.reid_engine.compare(embedding, stored)
        return self._cosine(embedding, stored)

    def match(self, embedding: np.ndarray, threshold: float = 0.55, exclude: set | None = None):
        """
        Retourne (nom, similarité) du bovin le plus proche, ou (None, sim_max).

        Version vectorisée: 1 BLAS sur toute la matrice au lieu d'une boucle Python.
        Fallback boucle pour N petit (overhead matrix > gain BLAS sous N≈10).

        exclude: ensemble de noms à IGNORER (utilisé pour la déduplication
        intra-frame: si Boeuf_001 est déjà attribué à un autre track dans
        la frame courante, on ne peut pas l'attribuer à celui-ci).
        """
        if not self.animals:
            return None, 0.0
        self._ensure_cache()
        if self._matrix is None or self._matrix.shape[0] == 0:
            return None, 0.0

        q = np.asarray(embedding, dtype=np.float32).ravel()
        if q.shape[0] != self._matrix.shape[1]:
            return None, 0.0

        n = self._matrix.shape[0]
        # Petit N: la boucle Python bat la vectorisation à cause de l'overhead.
        if n <= 10:
            return self._match_loop(q, threshold, exclude)

        if self.reid_engine is not None:
            sims = self._vectorized_compare(q)
        else:
            nq = float(np.linalg.norm(q))
            sims = (self._matrix @ q) / (self._norms * nq + 1e-8)

        if exclude:
            excl_mask = np.fromiter(
                (name in exclude for name in self._names),
                dtype=bool, count=n,
            )
            sims = np.where(excl_mask, -np.inf, sims)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= threshold:
            return self._names[best_idx], best_sim
        return None, best_sim

    def _match_loop(self, q: np.ndarray, threshold: float, exclude: set | None):
        """Fallback boucle pour N petit (<=10) où la vectorisation perd."""
        exclude = exclude or set()
        best_name, best_sim = None, -1.0
        for name in self._names:
            if name in exclude:
                continue
            stored = self.animals[name]["embedding"]
            sim = self._similarity(q, stored)
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim

    def _vectorized_compare(self, q: np.ndarray) -> np.ndarray:
        """
        Version batch de CattleReID.compare: cosine par composante pondérée.
        Évite la boucle Python sur N animaux (3 dot products par animal → 3 BLAS).
        """
        if self._matrix is None or self._matrix.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        D = self.reid_engine.DINO_DIM
        H = self.reid_engine.HSV_DIM
        L = self.reid_engine.LBP_DIM
        if q.shape[0] != D + H + L:
            # Fallback: cosine simple
            nq = float(np.linalg.norm(q))
            return (self._matrix @ q) / (self._norms * nq + 1e-8)

        q_d, q_h, q_l = q[:D], q[D:D + H], q[D + H:]
        M_d = self._matrix[:, :D]
        M_h = self._matrix[:, D:D + H]
        M_l = self._matrix[:, D + H:]

        n_d = np.linalg.norm(M_d, axis=1)
        n_h = np.linalg.norm(M_h, axis=1)
        n_l = np.linalg.norm(M_l, axis=1)
        nq_d = float(np.linalg.norm(q_d))
        nq_h = float(np.linalg.norm(q_h))
        nq_l = float(np.linalg.norm(q_l))

        sim_d = (M_d @ q_d) / (n_d * nq_d + 1e-8)
        sim_h = (M_h @ q_h) / (n_h * nq_h + 1e-8)
        sim_l = (M_l @ q_l) / (n_l * nq_l + 1e-8)
        w = self.reid_engine
        return (w.w_dino * sim_d + w.w_hsv * sim_h + w.w_lbp * sim_l).astype(np.float32)

    def add(self, name: str, embedding: np.ndarray) -> None:
        self.animals[name] = {
            "embedding": np.asarray(embedding, dtype=np.float32).ravel(),
            "count": 1,
            "first_seen": datetime.now().isoformat(timespec="seconds"),
        }
        self._dirty = True

    def rename(self, old: str, new: str) -> bool:
        if old in self.animals and new not in self.animals:
            self.animals[new] = self.animals.pop(old)
            self._dirty = True
            return True
        return False

    def update(self, name: str, embedding: np.ndarray, alpha: float = 0.2) -> None:
        """Mise à jour par moyenne mobile exponentielle."""
        if name in self.animals:
            old = self.animals[name]["embedding"]
            new = (1.0 - alpha) * old + alpha * embedding
            new = new / (np.linalg.norm(new) + 1e-8)
            self.animals[name]["embedding"] = new.astype(np.float32)
            self.animals[name]["count"] += 1
            self._dirty = True