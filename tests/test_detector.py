"""Tests detector.py — deduplication des boites et association des pistes.

Ces deux briques conditionnent directement la qualite du comptage :
un doublon = un bovin fantome en base ; un echange de piste = un nom qui
saute d'un animal a l'autre.
"""
import numpy as np
import pytest

from detector import (
    CATTLE_CLASS_IDS, COW_CLASS_ID, _SimpleIoUTracker, _iou_matrix, dedup_boxes,
)


# ─────────────────────────────────────────────────────────────
#  dedup_boxes
# ─────────────────────────────────────────────────────────────
def test_dedup_empty():
    assert len(dedup_boxes(np.empty((0, 4)), np.array([]))) == 0


def test_dedup_keeps_distinct_boxes():
    xyxy = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    keep = dedup_boxes(xyxy, np.array([0.9, 0.8]))
    assert keep.tolist() == [0, 1]


def test_dedup_drops_nested_box():
    """Une boite quasi contenue dans une autre est un doublon, pas un veau."""
    grande = [0, 0, 100, 100]
    incluse = [5, 5, 90, 90]        # ~72 % de la grande, 100 % d'elle-meme
    keep = dedup_boxes(np.array([grande, incluse], dtype=np.float32),
                       np.array([0.9, 0.4]))
    assert keep.tolist() == [0]     # la plus sure est conservee


def test_dedup_keeps_small_animal_next_to_big_one():
    """Un veau a cote d'une vache n'est PAS contenu dedans -> conserve."""
    vache = [0, 0, 100, 100]
    veau = [105, 60, 135, 100]
    keep = dedup_boxes(np.array([vache, veau], dtype=np.float32),
                       np.array([0.9, 0.5]))
    assert keep.tolist() == [0, 1]


def test_dedup_keeps_highest_confidence():
    a = [0, 0, 100, 100]
    b = [2, 2, 98, 98]
    keep = dedup_boxes(np.array([a, b], dtype=np.float32), np.array([0.3, 0.95]))
    assert keep.tolist() == [1]


def test_dedup_drops_high_iou_duplicate():
    a = [0, 0, 100, 100]
    b = [3, 3, 103, 103]            # IoU ~0.88, aucune n'est contenue
    keep = dedup_boxes(np.array([a, b], dtype=np.float32), np.array([0.9, 0.8]))
    assert keep.tolist() == [0]


# ─────────────────────────────────────────────────────────────
#  Classes acceptees
# ─────────────────────────────────────────────────────────────
def test_cattle_classes_include_cow():
    assert COW_CLASS_ID in CATTLE_CLASS_IDS
    assert len(CATTLE_CLASS_IDS) == 3     # cow + horse + sheep


# ─────────────────────────────────────────────────────────────
#  _iou_matrix
# ─────────────────────────────────────────────────────────────
def test_iou_matrix_shape_and_values():
    dets = np.array([[0, 0, 10, 10]], dtype=np.float32)
    trks = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    m = _iou_matrix(dets, trks)
    assert m.shape == (1, 2)
    assert m[0, 0] == pytest.approx(1.0)
    assert m[0, 1] == pytest.approx(0.0)


def test_iou_matrix_empty():
    assert _iou_matrix(np.empty((0, 4)), np.empty((0, 4))).shape == (0, 0)


# ─────────────────────────────────────────────────────────────
#  Tracker
# ─────────────────────────────────────────────────────────────
def test_tracker_assigns_new_ids():
    t = _SimpleIoUTracker()
    ids = t.update(np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32))
    assert sorted(ids.tolist()) == [1, 2]


def test_tracker_keeps_id_across_frames():
    t = _SimpleIoUTracker()
    first = t.update(np.array([[0, 0, 10, 10]], dtype=np.float32))
    second = t.update(np.array([[1, 1, 11, 11]], dtype=np.float32))
    assert first.tolist() == second.tolist()


def test_tracker_recovers_id_after_short_occlusion():
    """Un bovin masque puis reapparu doit RETROUVER son identifiant.

    Sinon il repasse par la re-identification et risque un nouveau nom.
    """
    t = _SimpleIoUTracker(max_lost=45)
    box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    original = t.update(box)[0]
    for _ in range(5):
        t.update(np.empty((0, 4), dtype=np.float32))   # occlusion totale
    assert t.update(box)[0] == original


def test_tracker_drops_track_after_max_lost():
    t = _SimpleIoUTracker(max_lost=3)
    box = np.array([[0, 0, 10, 10]], dtype=np.float32)
    original = t.update(box)[0]
    for _ in range(6):
        t.update(np.empty((0, 4), dtype=np.float32))
    assert t.update(box)[0] != original


def test_tracker_does_not_swap_neighbours():
    """Deux bovins cote a cote qui se croisent legerement gardent leur ID.

    C'est le cas que l'appariement glouton ratait : la premiere detection
    prenait la piste du voisin des lors qu'elle la recouvrait un peu.
    """
    t = _SimpleIoUTracker()
    a = [0.0, 0.0, 100.0, 100.0]
    b = [80.0, 0.0, 180.0, 100.0]          # chevauche a sur 20 %
    ids0 = t.update(np.array([a, b], dtype=np.float32))
    # Les deux avancent de 10 px vers la droite
    a2 = [10.0, 0.0, 110.0, 100.0]
    b2 = [90.0, 0.0, 190.0, 100.0]
    ids1 = t.update(np.array([a2, b2], dtype=np.float32))
    assert ids1.tolist() == ids0.tolist()


def test_tracker_handles_empty_frames():
    t = _SimpleIoUTracker()
    assert len(t.update(np.empty((0, 4), dtype=np.float32))) == 0


def test_tracker_never_reuses_an_id_within_a_frame():
    t = _SimpleIoUTracker()
    boxes = np.array([[0, 0, 10, 10], [2, 2, 12, 12], [4, 4, 14, 14]],
                     dtype=np.float32)
    t.update(boxes)
    ids = t.update(boxes)
    assert len(set(ids.tolist())) == len(ids)
