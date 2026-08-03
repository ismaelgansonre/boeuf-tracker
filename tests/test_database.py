"""Tests EmbeddingDatabase — add/match/rename/persist/dim purge."""
import os
import pickle
import numpy as np
import pytest

from database import EmbeddingDatabase


class FakeReID:
    """Reproduit l'interface consommée par EmbeddingDatabase."""
    DEEP_DIM = 8
    HSV_DIM = 4
    LBP_DIM = 4
    TOTAL_DIM = 16
    w_deep = 0.5
    w_hsv = 0.3
    w_lbp = 0.2

    def slice_deep(self): return slice(0, self.DEEP_DIM)
    def slice_hsv(self):  return slice(self.DEEP_DIM, self.DEEP_DIM + self.HSV_DIM)
    def slice_lbp(self):  return slice(self.DEEP_DIM + self.HSV_DIM, self.TOTAL_DIM)

    @staticmethod
    def compare(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


@pytest.fixture
def db(tmp_path):
    return EmbeddingDatabase(path=str(tmp_path / "test.pkl"), reid_engine=FakeReID())


def _emb(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(16).astype(np.float32)
    return v / np.linalg.norm(v)


def test_db_empty_match_returns_none(db):
    name, sim = db.match(_emb(1))
    assert name is None
    assert sim == 0.0


def test_db_add_and_match_self(db):
    e = _emb(42)
    db.add("Marguerite", e)
    name, sim = db.match(e, threshold=0.9)
    assert name == "Marguerite"
    assert sim >= 0.99


def test_db_match_below_threshold_returns_none(db):
    db.add("A", _emb(1))
    name, _ = db.match(_emb(9999), threshold=0.99)
    assert name is None


def test_db_match_exclude(db):
    e = _emb(3)
    db.add("A", e)
    db.add("B", _emb(4))
    # Sans exclude, A gagne (identique). Avec A exclu, on retombe sur B (mais < threshold).
    name, _ = db.match(e, threshold=0.5, exclude={"A"})
    assert name != "A"


def test_db_rename(db):
    db.add("Old", _emb(1))
    assert db.rename("Old", "New")
    assert "New" in db.animals and "Old" not in db.animals


def test_db_persist_roundtrip(tmp_path):
    path = tmp_path / "roundtrip.pkl"
    db1 = EmbeddingDatabase(path=str(path), reid_engine=FakeReID())
    db1.add("A", _emb(7))
    db1.save()
    db2 = EmbeddingDatabase(path=str(path), reid_engine=FakeReID())
    assert "A" in db2.animals
    assert db2.animals["A"]["embedding"].shape == (16,)


def test_db_validate_dim_purges_mismatched(db):
    db.add("good", _emb(1))
    # Injecte manuellement un embed de mauvaise dim
    db.animals["bad"] = {
        "embedding": np.zeros(999, dtype=np.float32),
        "count": 1, "first_seen": "",
    }
    remaining = db.validate_dim(expected_dim=16)
    assert remaining == 1
    assert "bad" not in db.animals


def test_db_vectorized_matches_loop_for_large_n(db):
    """Sanity : vectorisé et boucle Python donnent le même gagnant."""
    for i in range(20):  # >10 → chemin vectorisé
        db.add(f"a{i}", _emb(i))
    q = _emb(5)
    name, _ = db.match(q, threshold=0.0)
    assert name == "a5"
