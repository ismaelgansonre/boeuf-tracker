"""
behavior.py
-----------
Classification de comportement des bovins + suivi d'événements.

Extraction OOP de la logique qui vivait dans processor.py. Trois responsabilités
distinctes, chacune dans sa classe testable :

- BehaviorClassifier   : classe pure, mappe (vitesse, aspect, immobile_dur) → action
- BehaviorAnalyzer     : combine tracking history + classifieur + correction géo
- EventJournal         : file d'événements haut niveau (arrivée / comportement / départ)
- IsolationMonitor     : détecte les bovins isolés du troupeau > N sec
- TransitionMonitor    : détecte les changements de comportement par bovin

Toutes ces classes prennent leurs dépendances en paramètre — pas de globals,
pas de couplage à STATE. Le côté "state world" est passé explicitement par
processor.py.
"""
from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

try:
    from numba import njit as _njit
    NUMBA_OK = True
except Exception:  # pragma: no cover - fallback
    NUMBA_OK = False

    def _njit(*args, **kwargs):
        def deco(fn): return fn
        return deco


# ─────────────────────────────────────────────────────────────
#  Numba hot loops (extraites, testables en isolation)
# ─────────────────────────────────────────────────────────────
@_njit(cache=True, fastmath=True)
def total_displacement(pts_x: np.ndarray, pts_y: np.ndarray) -> float:
    """Somme des distances euclidiennes entre points consécutifs."""
    s = 0.0
    for i in range(1, pts_x.shape[0]):
        dx = pts_x[i] - pts_x[i - 1]
        dy = pts_y[i] - pts_y[i - 1]
        s += (dx * dx + dy * dy) ** 0.5
    return s


@_njit(cache=True, fastmath=True)
def classify_action_code(speed: float, aspect: float, rel_y: float, immobile_dur: float) -> int:
    """0=couché, 1=pâture, 2=boit, 3=immobile, 4=marche, 5=court, 6=rué.

    `speed` en largeurs-de-corps / seconde (invariant à la résolution/distance).

    Note: la détection "pâture" par aspect ratio seul est peu fiable — quand
    la vache baisse la tête, sa bbox s'allonge verticalement et aspect chute.
    La vraie signature (head_down via masque) est appliquée en aval par
    BehaviorAnalyzer, qui override "immobile" → "pâture" si head_down=True.
    """
    if aspect > 1.7 and speed < 0.05 and immobile_dur > 3.0:
        return 0
    if speed < 0.05 and aspect > 1.3 and rel_y > 0.6:
        return 2
    if speed < 0.10:
        return 3
    if speed < 0.60:
        return 4
    if speed < 1.50:
        return 5
    return 6


BEHAVIOR_LABELS: tuple[str, ...] = (
    "couché", "pâture", "boit", "immobile", "marche", "court", "rué",
)


# ─────────────────────────────────────────────────────────────
#  Classifieur pur (facile à unit-tester)
# ─────────────────────────────────────────────────────────────
@dataclass
class BehaviorClassifier:
    """Classifie une action à partir de features géométriques.

    Aucun état — pure fonction packagée en classe pour l'extensibilité
    (fine-tuning des seuils, injection de dépendances dans les tests).
    """
    def classify(
        self,
        speed_bws: float,
        aspect: float,
        rel_y: float,
        immobile_dur: float,
    ) -> str:
        code = classify_action_code(
            float(speed_bws), float(aspect), float(rel_y), float(immobile_dur),
        )
        return BEHAVIOR_LABELS[code]


# ─────────────────────────────────────────────────────────────
#  Analyzer : combine track history + classifieur + corrections
# ─────────────────────────────────────────────────────────────
@dataclass
class BehaviorAnalyzer:
    """Consomme (boxes, track_ids, timestamp) et produit une liste `behaviors`.

    Le track history est passé par référence (dict externe) — la même instance
    que celle utilisée par le reste du pipeline (STATE["track_history"]).

    Lissage temporel : chaque tid maintient une fenêtre glissante des N
    dernières actions ; l'action retournée est le vote majoritaire de la
    fenêtre. Élimine le flicker inter-frame (une frame de bruit → 1 changement
    d'étiquette éphémère). Recommandation littérature (CAMLLA-YOLO) : N=7-10.
    """
    track_history: dict[int, list[tuple[float, float, float]]]
    track_names: dict[int, str] = field(default_factory=dict)
    head_down_flags: dict[int, bool] = field(default_factory=dict)
    classifier: BehaviorClassifier = field(default_factory=BehaviorClassifier)
    history_window_s: float = 5.0
    immobile_disp_threshold_bw: float = 0.30
    smoothing_window: int = 7  # frames — vote majoritaire
    _action_history: dict[int, deque] = field(default_factory=dict)

    def _smooth(self, tid: int, action: str) -> str:
        """Retourne l'action majoritaire sur la fenêtre glissante pour ce tid."""
        buf = self._action_history.get(tid)
        if buf is None or buf.maxlen != self.smoothing_window:
            buf = deque(maxlen=self.smoothing_window)
            self._action_history[tid] = buf
        buf.append(action)
        # majority vote — ties broken by most recent (Counter est stable)
        return Counter(buf).most_common(1)[0][0]

    def forget_track(self, tid: int) -> None:
        """Libère la fenêtre de lissage d'un track disparu."""
        self._action_history.pop(tid, None)

    def analyze(
        self,
        boxes: np.ndarray | list,
        track_ids: Iterable[int],
        t_now: float,
        frame_shape: tuple[int, int] | None = None,
    ) -> list[dict]:
        behaviors: list[dict] = []
        frame_h = frame_shape[0] if frame_shape is not None else 1080

        for box, tid in zip(boxes, track_ids):
            tid = int(tid)
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            bw = max(box[2] - box[0], 1.0)
            bh = max(box[3] - box[1], 1.0)
            aspect = bw / bh
            rel_y = cy / frame_h

            hist = self.track_history.setdefault(tid, [])
            hist.append((cx, cy, t_now))
            # Prune old points
            self.track_history[tid] = [
                p for p in hist if t_now - p[2] < self.history_window_s
            ]
            pts = self.track_history[tid]
            if len(pts) < 3:
                continue

            dx = pts[-1][0] - pts[0][0]
            dy = pts[-1][1] - pts[0][1]
            dist = (dx * dx + dy * dy) ** 0.5
            dt = max(pts[-1][2] - pts[0][2], 1e-3)
            speed = (dist / dt) / bw  # body-widths / s

            pts_arr = np.asarray(pts, dtype=np.float32)
            total_disp = total_displacement(pts_arr[:, 0], pts_arr[:, 1]) / bw
            immobile_dur = (t_now - pts[0][2]) if total_disp < self.immobile_disp_threshold_bw else 0.0

            action = self.classifier.classify(speed, aspect, rel_y, immobile_dur)

            # Correction géométrique : tête baissée + quasi-immobile → pâture
            if self.head_down_flags.get(tid, False) and speed < 0.60:
                action = "pâture"

            # Lissage temporel (vote majoritaire sur N frames)
            action = self._smooth(tid, action)

            behaviors.append({
                "name": self.track_names.get(tid, "?"),
                "action": action,
                "speed": round(float(speed), 1),
                "track_id": tid,
                "aspect": round(float(aspect), 2),
            })
        return behaviors


# ─────────────────────────────────────────────────────────────
#  Journal d'événements
# ─────────────────────────────────────────────────────────────
@dataclass
class EventJournal:
    """File circulaire d'événements haut niveau.

    Injection : le buffer d'événements (STATE['events']) est fourni de l'extérieur
    pour rester compatible avec le reste de l'app qui expose STATE en JSON.
    """
    events_ref: list
    max_events: int = 40

    def push(self, kind: str, text: str, name: str | None = None) -> None:
        self.events_ref.insert(0, {
            "kind": kind, "text": text, "name": name, "ts": time.time(),
        })
        del self.events_ref[self.max_events:]


# ─────────────────────────────────────────────────────────────
#  Détection d'isolement (bovin loin du centroïde du troupeau)
# ─────────────────────────────────────────────────────────────
@dataclass
class IsolationMonitor:
    journal: EventJournal
    dist_frac_threshold: float = 0.35
    min_duration_s: float = 6.0
    _isolated_since: dict[str, float] = field(default_factory=dict)
    _flagged: set[str] = field(default_factory=set)

    def update(
        self,
        boxes: np.ndarray | list,
        track_ids: Iterable[int],
        frame_shape: tuple[int, int, int],
        track_names: dict[int, str],
        now: float | None = None,
    ) -> None:
        now = now if now is not None else time.time()
        if boxes is None or len(boxes) < 3:
            self._isolated_since.clear()
            return

        fh, fw = frame_shape[:2]
        diag = (fw * fw + fh * fh) ** 0.5
        centers = np.array([
            ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes
        ])
        troop = centers.mean(axis=0)

        active_names: set[str] = set()
        for c, tid in zip(centers, track_ids):
            name = track_names.get(int(tid))
            if not name or name == "?":
                continue
            active_names.add(name)
            dist_frac = float(np.linalg.norm(c - troop)) / diag
            if dist_frac > self.dist_frac_threshold:
                self._isolated_since.setdefault(name, now)
                if (name not in self._flagged
                        and now - self._isolated_since[name] > self.min_duration_s):
                    self._flagged.add(name)
                    self.journal.push("alert", f"{name} s'est isole du troupeau", name=name)
            else:
                self._isolated_since.pop(name, None)
                self._flagged.discard(name)

        for gone in list(self._isolated_since.keys()):
            if gone not in active_names:
                self._isolated_since.pop(gone, None)
                self._flagged.discard(gone)


# ─────────────────────────────────────────────────────────────
#  Détection de transitions de comportement + départ
# ─────────────────────────────────────────────────────────────
_BEHAVIOR_VERB: dict[str, str] = {
    "pâture":   "se met a paturer",
    "boit":     "commence a boire",
    "couché":   "se couche",
    "immobile": "s'arrete",
    "marche":   "se met en marche",
    "court":    "se met a courir",
    "rué":      "charge",
}


@dataclass
class TransitionMonitor:
    journal: EventJournal
    departure_grace_s: float = 8.0
    _last_behavior: dict[str, str] = field(default_factory=dict)
    _last_seen: dict[str, float] = field(default_factory=dict)

    def update(self, behaviors: list[dict], now: float | None = None) -> None:
        now = now if now is not None else time.time()
        seen_now: set[str] = set()

        for b in behaviors:
            name = b.get("name")
            action = b.get("action")
            if not name or name == "?" or not action:
                continue
            seen_now.add(name)
            self._last_seen[name] = now
            if self._last_behavior.get(name) != action:
                self._last_behavior[name] = action
                verb = _BEHAVIOR_VERB.get(action, f"passe en {action}")
                self.journal.push("behavior", f"{name} {verb}", name=name)

        stale = [n for n, t in self._last_seen.items()
                 if n not in seen_now and now - t > self.departure_grace_s]
        for n in stale:
            self._last_seen.pop(n, None)
            self._last_behavior.pop(n, None)
            self.journal.push("departure", f"{n} a quitte le champ", name=n)
