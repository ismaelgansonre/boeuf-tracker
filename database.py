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


class EmbeddingDatabase:
    """Base simple de vecteurs d'embeddings par bovin."""

    def __init__(self, path: str = "cattle_db.pkl"):
        self.path = path
        self.animals: dict = {}
        self.load()

    def load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    self.animals = pickle.load(f)
                print(f"[DB] {len(self.animals)} animaux chargés depuis {self.path}")
            except Exception as e:
                print(f"[DB] Erreur de lecture ({e}), nouvelle base.")
                self.animals = {}
        else:
            print(f"[DB] Nouvelle base (fichier {self.path} sera créé)")

    def save(self) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(self.animals, f)
        print(f"[DB] {len(self.animals)} animaux sauvegardés dans {self.path}")

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def match(self, embedding: np.ndarray, threshold: float = 0.55):
        """Retourne (nom, similarité) du bovin le plus proche, ou (None, sim_max)."""
        if not self.animals:
            return None, 0.0
        best_name, best_sim = None, -1.0
        for name, data in self.animals.items():
            sim = self._cosine(embedding, data["embedding"])
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