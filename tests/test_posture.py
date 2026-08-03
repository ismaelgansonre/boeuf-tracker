"""Tests posture.py — détection tête-au-sol sur masques synthétiques.

Les masques reproduisent la géométrie réelle vue de profil (le cas qui
faisait échouer l'heuristique par aspect ratio) :

  DEBOUT          : corps + 4 sabots fins qui touchent le sol
  PÂTURE          : corps + 4 sabots + mufle large posé au sol à une extrémité
"""
import numpy as np
import pytest

from posture import HeadMotionTracker, HeadState, MaskHeadAnalyzer, longest_run


# ─────────────────────────────────────────────────────────────
#  Générateurs de masques synthétiques
# ─────────────────────────────────────────────────────────────
def _standing_mask(w=300, h=200):
    """Vache debout de profil : bbox LARGE (aspect 1.5), sabots fins au sol."""
    m = np.zeros((h, w), dtype=bool)
    m[int(h * 0.15):int(h * 0.60), int(w * 0.08):int(w * 0.92)] = True  # corps
    leg_w = int(w * 0.05)
    for lx in (0.14, 0.26, 0.68, 0.82):  # 4 pattes
        x = int(w * lx)
        m[int(h * 0.55):h, x:x + leg_w] = True
    return m


def _grazing_mask(w=300, h=200, head_left=True):
    """Vache qui broute de profil : MÊME bbox large, + mufle au sol.

    C'est exactement le cas Colette/Aurelius : aspect ratio identique à
    `_standing_mask`, seule la géométrie au ras du sol diffère.
    """
    m = _standing_mask(w, h)
    head_w = int(w * 0.20)  # mufle : bien plus large qu'un sabot
    x0 = int(w * 0.02) if head_left else w - head_w - int(w * 0.02)
    m[int(h * 0.55):h, x0:x0 + head_w] = True          # tête/mufle au sol
    m[int(h * 0.30):int(h * 0.60), x0:x0 + head_w] = True  # encolure
    return m


@pytest.fixture
def analyzer():
    return MaskHeadAnalyzer()


# ─────────────────────────────────────────────────────────────
#  longest_run
# ─────────────────────────────────────────────────────────────
def test_longest_run_empty():
    assert longest_run(np.zeros(10, dtype=bool)) == (0, -1)


def test_longest_run_picks_longest():
    flags = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
    length, center = longest_run(flags)
    assert length == 4
    assert 3 <= center <= 6


def test_longest_run_full():
    length, _ = longest_run(np.ones(7, dtype=bool))
    assert length == 7


# ─────────────────────────────────────────────────────────────
#  Le cas qui échouait : profil large
# ─────────────────────────────────────────────────────────────
def test_standing_profile_is_not_head_down(analyzer):
    state = analyzer.analyze(_standing_mask())
    assert state.head_down is False
    assert state.max_run_frac < 0.13


def test_grazing_profile_is_head_down(analyzer):
    """Régression : bbox aussi large que debout, doit quand même détecter."""
    state = analyzer.analyze(_grazing_mask())
    assert state.head_down is True
    assert state.confidence > 0.5


def test_grazing_detected_on_both_sides(analyzer):
    for side in (True, False):
        assert analyzer.analyze(_grazing_mask(head_left=side)).head_down is True


def test_standing_and_grazing_share_aspect_ratio():
    """Prouve que l'aspect ratio ne peut PAS discriminer ces deux cas."""
    a, b = _standing_mask(), _grazing_mask()
    assert a.shape == b.shape  # même bbox → même aspect → signal inutile


def test_grazing_head_x_is_lateral(analyzer):
    left = analyzer.analyze(_grazing_mask(head_left=True))
    right = analyzer.analyze(_grazing_mask(head_left=False))
    assert left.head_x_frac < 0.4
    assert right.head_x_frac > 0.6


# ─────────────────────────────────────────────────────────────
#  Robustesse
# ─────────────────────────────────────────────────────────────
def test_empty_mask_is_unknown(analyzer):
    assert analyzer.analyze(np.zeros((0, 0), dtype=bool)).head_down is False
    assert analyzer.analyze(None).head_down is False


def test_tiny_mask_is_unknown(analyzer):
    assert analyzer.analyze(np.ones((6, 6), dtype=bool)).head_down is False


def test_accepts_uint8_mask(analyzer):
    assert analyzer.analyze(_grazing_mask().astype(np.uint8)).head_down is True


def test_scale_invariant(analyzer):
    """Même verdict à 2 résolutions différentes (bovin proche vs loin)."""
    small = analyzer.analyze(_grazing_mask(w=150, h=100))
    large = analyzer.analyze(_grazing_mask(w=600, h=400))
    assert small.head_down == large.head_down is True


# ─────────────────────────────────────────────────────────────
#  HeadMotionTracker
# ─────────────────────────────────────────────────────────────
def _state(down: bool, x: float = 0.2) -> HeadState:
    return HeadState(down, 0.8 if down else 0.0, 0.2 if down else 0.05, x, "test")


def test_motion_requires_persistence():
    t = HeadMotionTracker(window=10, min_ratio=0.5)
    # une seule frame tête basse sur 10 → pas de pâture
    t.update(1, _state(True))
    for _ in range(9):
        t.update(1, _state(False))
    assert t.is_grazing(1) is False


def test_motion_confirms_sustained_grazing():
    t = HeadMotionTracker(window=10, min_ratio=0.5)
    for _ in range(8):
        t.update(1, _state(True))
    assert t.is_grazing(1) is True


def test_motion_absorbs_isolated_false_negative():
    t = HeadMotionTracker(window=10, min_ratio=0.5)
    for i in range(10):
        t.update(1, _state(i != 5))  # 9 vrais, 1 raté
    assert t.is_grazing(1) is True


def test_motion_sweep_measures_lateral_movement():
    t = HeadMotionTracker(window=10)
    for x in (0.10, 0.20, 0.30, 0.40):
        t.update(1, _state(True, x=x))
    assert t.sweep(1) > 0.05          # mufle qui balaye
    t2 = HeadMotionTracker(window=10)
    for _ in range(4):
        t2.update(2, _state(True, x=0.25))
    assert t2.sweep(2) == pytest.approx(0.0, abs=1e-5)  # tête fixe


def test_motion_prune_frees_lost_tracks():
    t = HeadMotionTracker()
    t.update(1, _state(True))
    t.update(2, _state(True))
    t.prune({1})
    assert t.is_grazing(1) is True
    assert t.is_grazing(2) is False
