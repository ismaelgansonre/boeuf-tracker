"""
database.py
-----------
Stocke et persiste les embeddings des bovins connus.
Format: pickle avec {nom: {embedding: np.array, count: int, first_seen: str}}
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
        self.load()

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

        exclude: ensemble de noms à IGNORER (utilisé pour la déduplication
        intra-frame: si Boeuf_001 est déjà attribué à un autre track dans
        la frame courante, on ne peut pas l'attribuer à celui-ci).
        """
        if not self.animals:
            return None, 0.0
        exclude = exclude or set()
        best_name, best_sim = None, -1.0
        for name, data in self.animals.items():
            if name in exclude:
                continue
            sim = self._similarity(embedding, data["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim

    def add(self, name: str, embedding: np.ndarray) -> None:
        self.animals[name] = {
            "embedding": embedding,
            "count": 1,
            "first_seen": datetime.now().isoformat(timespec="seconds"),
        }

    def rename(self, old: str, new: str) -> bool:
        if old in self.animals and new not in self.animals:
            self.animals[new] = self.animals.pop(old)
            return True
        return False

    def update(self, name: str, embedding: np.ndarray, alpha: float = 0.2) -> None:
        """Mise à jour par moyenne mobile exponentielle."""
        if name in self.animals:
            old = self.animals[name]["embedding"]
            new = (1.0 - alpha) * old + alpha * embedding
            new = new / (np.linalg.norm(new) + 1e-8)
            self.animals[name]["embedding"] = new
            self.animals[name]["count"] += 1