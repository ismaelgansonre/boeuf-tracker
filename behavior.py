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
def classify_action_code(
    speed: float,
    aspect: float,
    rel_y: float,
    immobile_dur: float,
    lying: bool = False,
    head_down: bool = False,
    ruminate_after_s: float = 25.0,
) -> int:
    """0=couché, 1=pâture, 2=boit, 3=immobile, 4=marche, 5=court, 6=rué,
    7=rumine.

    `speed` en largeurs-de-corps / seconde (invariant à la résolution/distance).

    ORDRE DE PRIORITÉ
    -----------------
    Les signaux issus du masque (`lying`, `head_down`) l'emportent sur la
    géométrie de la boîte : ils décrivent la silhouette réelle, alors que
    l'aspect ratio confond une vache couchée et une vache debout de profil.

    1. `lying`     → couché (ou rumine si l'immobilité dure).
    2. `head_down` → pâture (mufle au sol) tant que l'animal ne court pas.
    3. Repli géométrique pour les frames sans masque exploitable.

    Le cas corrigé : un bovin couché présente une large plage de masque au sol,
    ce qui levait `head_down` et le faisait étiqueter "pâture". `lying` est
    désormais testé en premier et neutralise `head_down` en amont (posture.py).

    "rumine" est un PROXY : on ne voit pas la mastication. On l'infère d'une
    immobilité prolongée (> `ruminate_after_s`) sans mufle au sol — la posture
    de rumination du bovin, couché comme debout.
    """
    if lying:
        if immobile_dur >= ruminate_after_s and not head_down:
            return 7
        return 0
    if head_down and speed < 0.60:
        # Mufle au sol tout en bas du cadre et strictement immobile : abreuvoir.
        # Seuil haut (0.85) : à 0.70, tout bovin broutant au premier plan
        # (fréquent en vidéo smartphone) était étiqueté "boit" sans eau.
        if rel_y > 0.85 and speed < 0.05:
            return 2
        return 1
    # ── Repli géométrique (pas de masque exploitable cette frame) ──
    if aspect > 1.7 and speed < 0.05 and immobile_dur > 3.0:
        return 0
    if speed < 0.05 and aspect > 1.3 and rel_y > 0.6:
        return 2
    # 0.10 -> 0.06 : la marche lente d'un bovin au paturage descend souvent
    # sous 0.10 largeur de corps/s et se faisait etiqueter "immobile".
    if speed < 0.06:
        if immobile_dur >= ruminate_after_s:
            return 7
        return 3
    if speed < 0.60:
        return 4
    if speed < 1.50:
        return 5
    return 6


BEHAVIOR_LABELS: tuple[str, ...] = (
    "couché", "pâture", "boit", "immobile", "marche", "court", "rué",
    "rumine",
)

# Actions de transition — déduites du CHANGEMENT de posture, pas d'un seuil
# instantané. Elles ne sortent donc pas de `classify_action_code`.
ACTION_LIE_DOWN = "se couche"
ACTION_STAND_UP = "se lève"

# Postures (orthogonales à l'action : on peut ruminer couché comme debout).
POSTURE_LYING = "couché"
POSTURE_STANDING = "debout"


# ─────────────────────────────────────────────────────────────
#  Classifieur pur (facile à unit-tester)
# ─────────────────────────────────────────────────────────────
@dataclass
class BehaviorClassifier:
    """Classifie une action à partir de features géométriques.

    Aucun état — pure fonction packagée en classe pour l'extensibilité
    (fine-tuning des seuils, injection de dépendances dans les tests).
    """
    ruminate_after_s: float = 25.0

    def classify(
        self,
        speed_bws: float,
        aspect: float,
        rel_y: float,
        immobile_dur: float,
        lying: bool = False,
        head_down: bool = False,
    ) -> str:
        code = classify_action_code(
            float(speed_bws), float(aspect), float(rel_y), float(immobile_dur),
            bool(lying), bool(head_down), float(self.ruminate_after_s),
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

    Postures et transitions : `lying_flags` (issu de posture.py) donne la
    posture courante. Un changement de posture produit une action transitoire
    ("se couche" / "se lève") affichée pendant `transition_hold_s` — c'est le
    moment le plus informatif pour l'éleveur (une vache qui peine à se lever
    est un signal sanitaire).
    """
    track_history: dict[int, list[tuple[float, float, float]]]
    track_names: dict[int, str] = field(default_factory=dict)
    head_down_flags: dict[int, bool] = field(default_factory=dict)
    lying_flags: dict[int, bool] = field(default_factory=dict)
    classifier: BehaviorClassifier = field(default_factory=BehaviorClassifier)
    history_window_s: float = 5.0
    immobile_disp_threshold_bw: float = 0.30
    max_jump_bw: float = 1.0   # largeurs de corps / frame avant reset d'historique
    min_speed_window_s: float = 0.5  # base temporelle mini pour estimer une vitesse
    smoothing_window: int = 7  # frames — vote majoritaire
    # Durée d'affichage d'une transition de posture. ~1.5 s correspond au temps
    # que met un bovin à se relever ; au-delà, l'étiquette reste affichée alors
    # que l'animal est déjà stabilisé dans sa nouvelle posture.
    transition_hold_s: float = 1.5
    # Réarmement minimal entre deux transitions de posture d'un même bovin.
    # Le drapeau `lying` peut osciller autour de son hystérésis (silhouette
    # partiellement fusionnée avec un congénère) : sans réarmement, chaque
    # oscillation redéclenche "se couche"/"se lève" et l'étiquette de
    # transition monopolise l'affichage pendant des dizaines de frames.
    # Un bovin ne se couche pas deux fois en 5 s.
    transition_cooldown_s: float = 5.0
    _action_history: dict[int, deque] = field(default_factory=dict)
    _last_lying: dict[int, bool] = field(default_factory=dict)
    _transition: dict[int, tuple[str, float]] = field(default_factory=dict)
    _immobile_since: dict[int, float] = field(default_factory=dict)
    _last_transition_t: dict[int, float] = field(default_factory=dict)
    # Compensation du mouvement de caméra (vidéo tenue à la main) :
    # centres de la frame précédente + décalage cumulé caméra.
    _prev_centers: dict[int, tuple[float, float]] = field(default_factory=dict)
    _cam_off: tuple[float, float] = (0.0, 0.0)

    def _smooth(self, tid: int, action: str) -> str:
        """Retourne l'action majoritaire sur la fenêtre glissante pour ce tid."""
        buf = self._action_history.get(tid)
        if buf is None or buf.maxlen != self.smoothing_window:
            buf = deque(maxlen=self.smoothing_window)
            self._action_history[tid] = buf
        buf.append(action)
        # majority vote — ties broken by most recent (Counter est stable)
        return Counter(buf).most_common(1)[0][0]

    def _immobile_duration(self, tid: int, still: bool, t_now: float) -> float:
        """Durée d'immobilité continue, en secondes, SANS plafond.

        `track_history` est élagué à `history_window_s` (5 s) : en déduire la
        durée d'immobilité la plafonnait mécaniquement à 5 s, ce qui rendait
        tout seuil supérieur (rumination) inatteignable. On mémorise donc
        l'instant où l'animal s'est arrêté, indépendamment de l'historique
        de positions.
        """
        if not still:
            self._immobile_since.pop(tid, None)
            return 0.0
        started = self._immobile_since.setdefault(tid, t_now)
        return max(0.0, t_now - started)

    def _posture_transition(self, tid: int, lying: bool, t_now: float) -> str | None:
        """Retourne "se couche"/"se lève" pendant `transition_hold_s` après un
        changement de posture, sinon None.

        La transition est mémorisée plutôt que déduite à la volée : elle ne dure
        qu'une frame dans le signal brut, alors qu'elle doit rester lisible à
        l'écran et dans le journal quelques secondes.
        """
        previous = self._last_lying.get(tid)
        self._last_lying[tid] = lying
        if previous is not None and previous != lying:
            # Réarmement : un flip survenant trop tôt après la dernière
            # transition est du bruit d'hystérésis, pas un vrai mouvement.
            last_t = self._last_transition_t.get(tid, -1e12)
            if t_now - last_t >= self.transition_cooldown_s:
                self._last_transition_t[tid] = t_now
                label = ACTION_LIE_DOWN if lying else ACTION_STAND_UP
                self._transition[tid] = (label, t_now)
                return label
        pending = self._transition.get(tid)
        if pending is not None:
            label, started = pending
            if t_now - started <= self.transition_hold_s:
                return label
            self._transition.pop(tid, None)
        return None

    def forget_track(self, tid: int) -> None:
        """Libère la fenêtre de lissage d'un track disparu."""
        self._action_history.pop(tid, None)
        self._last_lying.pop(tid, None)
        self._transition.pop(tid, None)
        self._immobile_since.pop(tid, None)
        self._last_transition_t.pop(tid, None)
        self._prev_centers.pop(tid, None)

    def analyze(
        self,
        boxes: np.ndarray | list,
        track_ids: Iterable[int],
        t_now: float,
        frame_shape: tuple[int, int] | None = None,
    ) -> list[dict]:
        behaviors: list[dict] = []
        frame_h = frame_shape[0] if frame_shape is not None else 1080

        # ── Compensation du mouvement de caméra ──────────────────────
        # Sur une vidéo tenue à la main (smartphone), un panoramique déplace
        # TOUTES les boîtes d'un même vecteur : chaque bovin immobile prend
        # alors une "vitesse" égale au mouvement de caméra et se fait
        # étiqueter "court"/"rué". La médiane des déplacements inter-frame
        # de toutes les pistes estime ce mouvement global (les animaux qui
        # bougent réellement sont minoritaires et la médiane les ignore) ;
        # on travaille ensuite en coordonnées "monde" stabilisées.
        centers_now: dict[int, tuple[float, float]] = {}
        for box, tid in zip(boxes, track_ids):
            centers_now[int(tid)] = (
                (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0,
            )
        deltas = [
            (c[0] - self._prev_centers[t][0], c[1] - self._prev_centers[t][1])
            for t, c in centers_now.items() if t in self._prev_centers
        ]
        # Moins de 3 pistes communes : estimation trop bruitée, on n'y touche pas.
        if len(deltas) >= 3:
            gdx = float(np.median([d[0] for d in deltas]))
            gdy = float(np.median([d[1] for d in deltas]))
            self._cam_off = (self._cam_off[0] + gdx, self._cam_off[1] + gdy)
        self._prev_centers = centers_now
        off_x, off_y = self._cam_off

        for box, tid in zip(boxes, track_ids):
            tid = int(tid)
            cx = (box[0] + box[2]) / 2 - off_x
            cy = (box[1] + box[3]) / 2 - off_y
            bw = max(box[2] - box[0], 1.0)
            bh = max(box[3] - box[1], 1.0)
            aspect = bw / bh
            # Echelle de normalisation des vitesses : la plus GRANDE dimension
            # de la boite. Vu de face, bw n'est que la largeur d'epaules :
            # normaliser par bw amplifiait le balancement de tete d'un bovin
            # immobile a l'auge jusqu'au seuil de "marche".
            scale = max(bw, bh)
            # rel_y = position DANS LE CADRE (abreuvoir au bas de l'image) :
            # toujours en coordonnées écran brutes, pas stabilisées.
            rel_y = ((box[1] + box[3]) / 2.0) / frame_h

            hist = self.track_history.setdefault(tid, [])
            # Saut invraisemblable : la piste a change d'animal (identifiant
            # repris apres occlusion, ou echange entre voisins). Conserver
            # l'historique produirait une vitesse enorme, donc un faux "court"
            # ou "rué" — l'artefact dominant sur les sequences de troupeau.
            # On repart de zero : mieux vaut quelques frames sans verdict qu'un
            # verdict faux.
            if hist:
                jump = ((cx - hist[-1][0]) ** 2 + (cy - hist[-1][1]) ** 2) ** 0.5
                if jump > self.max_jump_bw * scale:
                    hist = []
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
            dt = pts[-1][2] - pts[0][2]
            # Base temporelle trop courte : la vitesse est dominee par le
            # tremblement de la boite, pas par le deplacement de l'animal. A
            # 30 FPS, 3 points couvrent 0.1 s ; un jitter de 10 % de largeur de
            # corps donne alors 1.0 largeur/s, soit le seuil de "rué". C'est
            # ce qui produisait des rafales de "court"/"rué" sur des bovins
            # immobiles. On attend d'avoir assez de recul.
            if dt < self.min_speed_window_s:
                continue
            speed = (dist / max(dt, 1e-3)) / scale  # longueurs de corps / s

            pts_arr = np.asarray(pts, dtype=np.float32)
            total_disp = total_displacement(pts_arr[:, 0], pts_arr[:, 1]) / scale
            immobile_dur = self._immobile_duration(
                tid, total_disp < self.immobile_disp_threshold_bw, t_now,
            )

            lying = bool(self.lying_flags.get(tid, False))
            head_down = bool(self.head_down_flags.get(tid, False))

            action = self.classifier.classify(
                speed, aspect, rel_y, immobile_dur,
                lying=lying, head_down=head_down,
            )

            # Lissage temporel (vote majoritaire sur N frames)
            action = self._smooth(tid, action)

            # La transition de posture prime sur l'état lissé : elle est brève
            # par nature et le vote majoritaire l'effacerait.
            transition = self._posture_transition(tid, lying, t_now)
            if transition is not None:
                action = transition

            behaviors.append({
                "name": self.track_names.get(tid, "?"),
                "action": action,
                "posture": POSTURE_LYING if lying else POSTURE_STANDING,
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
    "pâture":    "se met a paturer",
    "boit":      "commence a boire",
    "couché":    "est couche",
    "rumine":    "rumine",
    "se couche": "se couche",
    "se lève":   "se releve",
    "immobile":  "s'arrete",
    "marche":    "se met en marche",
    "court":     "se met a courir",
    "rué":       "charge",
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
