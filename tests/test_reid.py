"""Tests reid.py — HSV/LBP extractors + composition (backbone mocké)."""
import numpy as np
import pytest

from reid import (
    CattleReID, HSVHistExtractor, LBPExtractor, ReIDWeights,
)


def _rand_crop(h=128, w=128, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


class MockDeep:
    """Backbone stub — évite le download de MegaDescriptor pendant les tests."""
    def __init__(self, dim=64, seed=42):
        self.dim = dim
        self._rng = np.random.default_rng(seed)

    def _det(self, crop):
        """Déterministe : hash du crop → embedding fixe (même crop → même vecteur)."""
        if crop is None or crop.size == 0:
            return None
        h = int(hash(crop.tobytes()) & 0xFFFFFFFF)
        rng = np.random.default_rng(h)
        v = rng.standard_normal(self.dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8)

    def extract(self, crop):
        return self._det(crop)

    def extract_batch(self, crops):
        return [self._det(c) for c in crops]


# ─── HSV extractor ──────────────────────────────────────────
def test_hsv_dim_and_norm():
    ext = HSVHistExtractor()
    v = ext.extract(_rand_crop())
    assert v.shape == (48,)
    assert v.sum() == pytest.approx(1.0, rel=1e-3)


def test_hsv_rejects_small_crop():
    ext = HSVHistExtractor()
    assert ext.extract(_rand_crop(10, 10)) is None
    assert ext.extract(None) is None


# ─── LBP extractor ──────────────────────────────────────────
def test_lbp_dim_and_norm():
    ext = LBPExtractor()
    v = ext.extract(_rand_crop())
    assert v.shape == (32,)
    assert np.linalg.norm(v) == pytest.approx(1.0, rel=1e-3)


# ─── ReIDWeights ────────────────────────────────────────────
def test_weights_normalize():
    w = ReIDWeights(1.0, 2.0, 1.0).normalized()
    assert w.deep + w.hsv + w.lbp == pytest.approx(1.0)


# ─── CattleReID composition ─────────────────────────────────
@pytest.fixture
def reid():
    return CattleReID(deep_extractor=MockDeep(dim=64))


def test_reid_dim_layout(reid):
    assert reid.DEEP_DIM == 64
    assert reid.HSV_DIM == 48
    assert reid.LBP_DIM == 32
    assert reid.TOTAL_DIM == 144
    assert reid.slice_deep() == slice(0, 64)
    assert reid.slice_hsv() == slice(64, 112)
    assert reid.slice_lbp() == slice(112, 144)


def test_reid_embedding_norm(reid):
    e = reid.get_embedding(_rand_crop())
    assert e.shape == (144,)
    assert np.linalg.norm(e) == pytest.approx(1.0, rel=1e-3)


def test_reid_identity_similarity(reid):
    e = reid.get_embedding(_rand_crop(seed=7))
    assert reid.compare(e, e) == pytest.approx(1.0, rel=1e-3)


def test_reid_deterministic(reid):
    crop = _rand_crop(seed=99)
    e1 = reid.get_embedding(crop)
    e2 = reid.get_embedding(crop)
    assert np.allclose(e1, e2)


def test_reid_batch_equals_single(reid):
    crops = [_rand_crop(seed=i) for i in range(4)]
    singles = [reid.get_embedding(c) for c in crops]
    batch = reid.get_embedding_batch(crops)
    for s, b in zip(singles, batch):
        assert np.allclose(s, b, atol=1e-5)


def test_reid_batch_handles_none(reid):
    out = reid.get_embedding_batch([_rand_crop(), None, _rand_crop()])
    assert out[0] is not None and out[1] is None and out[2] is not None


def test_reid_compare_shape_mismatch_returns_zero(reid):
    e = reid.get_embedding(_rand_crop())
    assert reid.compare(e, np.zeros(999, dtype=np.float32)) == 0.0
    assert reid.compare(None, e) == 0.0
