"""Tests unitaires du module behavior."""
import numpy as np
import pytest

from behavior import (
    BehaviorAnalyzer, BehaviorClassifier, EventJournal,
    IsolationMonitor, TransitionMonitor, total_displacement,
)


# ─── Numba pure fns ─────────────────────────────────────────
def test_total_displacement_zero_when_static():
    x = np.array([100.0] * 5, dtype=np.float32)
    y = np.array([200.0] * 5, dtype=np.float32)
    assert total_displacement(x, y) == pytest.approx(0.0)


def test_total_displacement_line():
    x = np.array([0.0, 3.0, 6.0], dtype=np.float32)
    y = np.array([0.0, 4.0, 8.0], dtype=np.float32)
    # deux segments de 5 chacun
    assert total_displacement(x, y) == pytest.approx(10.0, rel=1e-3)


# ─── Classifier ─────────────────────────────────────────────
@pytest.fixture
def clf():
    return BehaviorClassifier()


def test_classifier_run(clf):
    assert clf.classify(2.0, 1.5, 0.5, 0) == "rué"


def test_classifier_couche(clf):
    assert clf.classify(0.001, 2.0, 0.5, 10.0) == "couché"


def test_classifier_marche(clf):
    assert clf.classify(0.3, 1.5, 0.5, 0) == "marche"


# ─── Analyzer ───────────────────────────────────────────────
def test_analyzer_produces_behavior_after_3_frames():
    hist = {}
    a = BehaviorAnalyzer(track_history=hist, track_names={1: "Marguerite"})
    # 3 frames espacées de 0.5s, boîte quasi-immobile
    box = [100.0, 100.0, 300.0, 400.0]
    out1 = a.analyze([box], [1], 0.0, (600, 800))
    assert out1 == []  # pas assez d'historique
    a.analyze([box], [1], 0.5, (600, 800))
    out3 = a.analyze([box], [1], 1.0, (600, 800))
    assert len(out3) == 1
    assert out3[0]["name"] == "Marguerite"
    assert out3[0]["track_id"] == 1


def test_analyzer_head_down_overrides_to_pature():
    hist = {}
    a = BehaviorAnalyzer(
        track_history=hist, track_names={1: "X"}, head_down_flags={1: True},
    )
    box = [100.0, 100.0, 300.0, 400.0]
    for t in (0.0, 0.5, 1.0):
        out = a.analyze([box], [1], t, (600, 800))
    assert out[0]["action"] == "pâture"


def test_analyzer_smoothing_absorbs_flicker():
    """Un bruit d'1 frame ne doit pas changer l'étiquette majoritaire."""
    hist = {}
    a = BehaviorAnalyzer(
        track_history=hist, track_names={1: "X"}, smoothing_window=5,
    )
    # 4 frames immobile puis 1 frame de "marche" (bruit) → doit rester immobile
    still = [100.0, 100.0, 300.0, 400.0]      # aspect~0.67 → immobile
    fast  = [100.0, 100.0, 800.0, 200.0]      # gros déplacement
    a.analyze([still], [1], 0.0, (600, 800))
    a.analyze([still], [1], 0.5, (600, 800))
    a.analyze([still], [1], 1.0, (600, 800))
    a.analyze([still], [1], 1.5, (600, 800))
    out = a.analyze([fast], [1], 2.0, (600, 800))
    # vote majoritaire : 4 immobiles + 1 fast → immobile gagne encore
    assert out[0]["action"] == "immobile"


# ─── Couché / rumine / transitions ──────────────────────────
def test_classifier_lying_flag_wins_over_head_down(clf):
    """Régression : un bovin couché lève head_down (large plage au sol) et
    était étiqueté "pâture". `lying` doit primer."""
    assert clf.classify(0.0, 1.2, 0.4, 2.0, lying=True, head_down=True) == "couché"


def test_classifier_head_down_gives_pature(clf):
    assert clf.classify(0.0, 1.2, 0.4, 2.0, head_down=True) == "pâture"


def test_classifier_lying_long_immobility_is_rumination(clf):
    assert clf.classify(0.0, 1.2, 0.4, 60.0, lying=True) == "rumine"


def test_classifier_standing_long_immobility_is_rumination(clf):
    assert clf.classify(0.0, 1.2, 0.4, 60.0) == "rumine"


def test_classifier_short_immobility_stays_immobile(clf):
    assert clf.classify(0.0, 1.2, 0.4, 2.0) == "immobile"


def test_analyzer_lying_flag_produces_couche():
    a = BehaviorAnalyzer(track_history={}, track_names={1: "X"},
                         head_down_flags={1: True}, lying_flags={1: True})
    box = [100.0, 100.0, 400.0, 300.0]
    for t in (0.0, 0.5, 1.0):
        out = a.analyze([box], [1], t, (600, 800))
    assert out[0]["action"] == "couché"
    assert out[0]["posture"] == "couché"


def test_analyzer_reports_standing_posture():
    a = BehaviorAnalyzer(track_history={}, track_names={1: "X"})
    box = [100.0, 100.0, 300.0, 400.0]
    for t in (0.0, 0.5, 1.0):
        out = a.analyze([box], [1], t, (600, 800))
    assert out[0]["posture"] == "debout"


def test_analyzer_emits_lie_down_then_stand_up():
    lying = {}
    # transition_cooldown_s abaissé : on teste ici la mécanique de transition,
    # pas le réarmement anti-rebond (testé séparément).
    a = BehaviorAnalyzer(track_history={}, track_names={1: "X"},
                         lying_flags=lying, transition_hold_s=0.1,
                         transition_cooldown_s=1.0)
    box = [100.0, 100.0, 400.0, 300.0]
    for t in (0.0, 0.5, 1.0):
        a.analyze([box], [1], t, (600, 800))
    lying[1] = True
    assert a.analyze([box], [1], 1.5, (600, 800))[0]["action"] == "se couche"
    lying[1] = False
    assert a.analyze([box], [1], 3.0, (600, 800))[0]["action"] == "se lève"


def test_analyzer_transition_cooldown_suppresses_flapping():
    """Un drapeau `lying` qui oscille autour de l'hystérésis ne doit pas
    redéclencher "se couche"/"se lève" à chaque flip : une seule transition
    est émise par fenêtre de `transition_cooldown_s`."""
    lying = {}
    a = BehaviorAnalyzer(track_history={}, track_names={1: "X"},
                         lying_flags=lying, transition_hold_s=0.1,
                         transition_cooldown_s=5.0)
    box = [100.0, 100.0, 400.0, 300.0]
    for t in (0.0, 0.5, 1.0):
        a.analyze([box], [1], t, (600, 800))
    lying[1] = True
    assert a.analyze([box], [1], 1.5, (600, 800))[0]["action"] == "se couche"
    # Flips rapprochés (bruit) : aucune nouvelle transition ne doit sortir.
    lying[1] = False
    assert a.analyze([box], [1], 2.0, (600, 800))[0]["action"] != "se lève"
    lying[1] = True
    assert a.analyze([box], [1], 2.5, (600, 800))[0]["action"] != "se couche"


def test_analyzer_transition_survives_majority_smoothing():
    """La transition est brève : le vote majoritaire l'effacerait sans le
    maintien explicite de `transition_hold_s`."""
    lying = {}
    a = BehaviorAnalyzer(track_history={}, track_names={1: "X"},
                         lying_flags=lying, smoothing_window=7,
                         transition_hold_s=3.0)
    box = [100.0, 100.0, 400.0, 300.0]
    for t in (0.0, 0.5, 1.0):
        a.analyze([box], [1], t, (600, 800))
    lying[1] = True
    a.analyze([box], [1], 1.5, (600, 800))
    out = a.analyze([box], [1], 2.0, (600, 800))     # frame suivante
    assert out[0]["action"] == "se couche"


def test_immobile_duration_is_not_capped_by_history_window():
    """Régression : la durée d'immobilité était déduite de `track_history`,
    élagué à 5 s — aucun seuil de rumination n'était atteignable."""
    a = BehaviorAnalyzer(track_history={}, track_names={1: "X"},
                         history_window_s=5.0)
    box = [100.0, 100.0, 300.0, 400.0]
    out = None
    for t in np.arange(0.0, 40.0, 0.5):
        out = a.analyze([box], [1], float(t), (600, 800))
    assert out[0]["action"] == "rumine"


def test_analyzer_forget_track_frees_buffer():
    a = BehaviorAnalyzer(track_history={})
    box = [100.0, 100.0, 300.0, 400.0]
    for t in (0.0, 0.5, 1.0):
        a.analyze([box], [42], t, (600, 800))
    assert 42 in a._action_history
    a.forget_track(42)
    assert 42 not in a._action_history


# ─── EventJournal ───────────────────────────────────────────
def test_journal_pushes_and_caps():
    buf = []
    j = EventJournal(events_ref=buf, max_events=3)
    for i in range(5):
        j.push("system", f"ev-{i}")
    assert len(buf) == 3
    # LIFO : le dernier poussé est en tête
    assert buf[0]["text"] == "ev-4"


# ─── IsolationMonitor ───────────────────────────────────────
def test_isolation_below_threshold_animals_no_alert():
    buf = []
    j = EventJournal(events_ref=buf)
    mon = IsolationMonitor(journal=j)
    # 2 animaux seulement → pas de notion de troupeau
    boxes = [[0, 0, 100, 100], [50, 50, 150, 150]]
    mon.update(boxes, [1, 2], (600, 800, 3), {1: "A", 2: "B"}, now=0.0)
    assert buf == []


def test_isolation_triggers_alert_after_duration():
    buf = []
    j = EventJournal(events_ref=buf)
    mon = IsolationMonitor(
        journal=j, dist_frac_threshold=0.05, min_duration_s=1.0,
    )
    # 3 animaux : deux collés, un très loin
    boxes = [[0, 0, 10, 10], [5, 5, 15, 15], [500, 500, 510, 510]]
    names = {1: "A", 2: "B", 3: "Loner"}
    mon.update(boxes, [1, 2, 3], (600, 800, 3), names, now=0.0)
    mon.update(boxes, [1, 2, 3], (600, 800, 3), names, now=2.0)
    assert any(e["name"] == "Loner" and e["kind"] == "alert" for e in buf)


# ─── TransitionMonitor ─────────────────────────────────────
def test_transition_emits_on_action_change():
    buf = []
    j = EventJournal(events_ref=buf)
    mon = TransitionMonitor(journal=j)
    mon.update([{"name": "A", "action": "marche"}], now=0.0)
    mon.update([{"name": "A", "action": "marche"}], now=1.0)  # pas de change
    mon.update([{"name": "A", "action": "court"}], now=2.0)   # change → event
    kinds = [e["kind"] for e in buf]
    assert kinds.count("behavior") == 2  # marche + court


def test_transition_emits_departure_after_grace():
    buf = []
    j = EventJournal(events_ref=buf)
    mon = TransitionMonitor(journal=j, departure_grace_s=1.0)
    mon.update([{"name": "A", "action": "marche"}], now=0.0)
    mon.update([], now=5.0)  # A absent > 1s
    assert any(e["kind"] == "departure" and e["name"] == "A" for e in buf)
