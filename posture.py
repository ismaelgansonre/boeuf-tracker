"""
posture.py
----------
Détection de la posture "tête au sol" (pâture) depuis le masque de segmentation.

POURQUOI PAS L'ASPECT RATIO
---------------------------
Une vache qui broute vue DE PROFIL garde une bbox LARGE (corps long horizontal
+ tête qui descend devant). Son aspect ratio reste 1.5-1.8, identique à une
vache debout. Tout critère basé sur `bw/bh` échoue donc sur les vues latérales
— c'est le cas le plus fréquent en pâturage.

LE VRAI SIGNAL : LA LARGEUR AU RAS DU SOL
-----------------------------------------
Au niveau du sol (bas de la bbox), une vache debout ne présente que des SABOTS :
4 colonnes fines, chacune ~4-8 % de la largeur du corps.

Une vache qui broute pose son MUFLE au sol : un bloc plein et continu de
~12-30 % de la largeur du corps, situé à une extrémité latérale.

Le discriminant est donc la **plus longue plage horizontale continue de masque
dans la bande basse** (`max_run_frac`) :

    debout   → max_run_frac ≈ 0.05 - 0.10   (un sabot)
    pâture   → max_run_frac ≈ 0.15 - 0.35   (un mufle)

Ce signal est invariant à l'orientation, à la distance caméra et à la
résolution — contrairement à l'aspect ratio ou au "centroïde Y".

SIGNAUX SECONDAIRES
-------------------
- Latéralité : le mufle est à une extrémité (gauche/droite), pas au centre.
- Solidité : le mufle est un bloc plein ; deux sabots collés le sont moins.
- Persistance temporelle (HeadMotionTracker) : une vache qui broute garde la
  tête basse plusieurs secondes et la balaye latéralement. Un artefact d'une
  frame ne persiste pas.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# ─────────────────────────────────────────────────────────────
#  Résultat
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class HeadState:
    """Posture de tête déduite d'un masque de segmentation."""
    head_down: bool
    confidence: float          # 0..1
    max_run_frac: float        # largeur de la plus longue plage au sol / largeur bbox
    head_x_frac: float | None  # position horizontale du mufle (0=gauche, 1=droite)
    reason: str                # explication courte (debug / UI)

    @classmethod
    def unknown(cls, reason: str = "masque inexploitable") -> "HeadState":
        return cls(False, 0.0, 0.0, None, reason)


# ─────────────────────────────────────────────────────────────
#  Utilitaire : plus longue plage continue de True
# ─────────────────────────────────────────────────────────────
def longest_run(flags: np.ndarray) -> tuple[int, int]:
    """Retourne (longueur, index_centre) de la plus longue plage de True.

    Retourne (0, -1) si aucun True. Vectorisé (pas de boucle Python).
    """
    if flags.size == 0 or not flags.any():
        return 0, -1
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    lengths = ends - starts
    i = int(np.argmax(lengths))
    return int(lengths[i]), int((starts[i] + ends[i] - 1) // 2)


# ─────────────────────────────────────────────────────────────
#  Analyseur principal
# ─────────────────────────────────────────────────────────────
@dataclass
class MaskHeadAnalyzer:
    """Déduit `head_down` de la géométrie du masque au ras du sol.

    Paramètres réglables (valeurs par défaut calibrées sur du pâturage en
    plein champ, vue latérale, bovins de 150-400 px de large) :

    band_frac        : hauteur de la bande basse analysée (fraction de la bbox).
                       0.18 = les 18 % du bas — sabots et mufle uniquement.
    run_threshold    : max_run_frac au-delà duquel on parle de mufle.
                       0.13 sépare bien un sabot (~0.07) d'un mufle (~0.20).
    strong_run       : au-delà, on conclut tête basse même sans latéralité.
    lateral_zone     : fraction extérieure considérée comme "extrémité".
                       0.40 → les 40 % gauche ou les 40 % droite.
    min_mask_px      : masque trop petit → on ne conclut pas.
    """
    band_frac: float = 0.18
    run_threshold: float = 0.13
    strong_run: float = 0.22
    lateral_zone: float = 0.40
    min_mask_px: int = 200

    def analyze(self, mask_roi: np.ndarray) -> HeadState:
        """mask_roi : masque binaire (bool/uint8) recadré sur la bbox du bovin."""
        if mask_roi is None or mask_roi.size == 0:
            return HeadState.unknown("masque vide")

        m = mask_roi.astype(bool)
        h, w = m.shape[:2]
        if h < 12 or w < 12 or m.sum() < self.min_mask_px:
            return HeadState.unknown("masque trop petit")

        # Bande basse : uniquement ce qui touche (presque) le sol
        band_start = int(h * (1.0 - self.band_frac))
        band = m[band_start:, :]
        if band.size == 0 or not band.any():
            return HeadState.unknown("bande basse vide")

        # Plus longue plage horizontale continue au ras du sol
        cols = band.any(axis=0)
        run_len, run_center = longest_run(cols)
        max_run_frac = run_len / w

        if run_center < 0:
            return HeadState.unknown("aucune plage au sol")

        head_x_frac = run_center / w
        # Latéralité : le mufle est à une extrémité, pas au milieu du corps
        is_lateral = (
            head_x_frac <= self.lateral_zone
            or head_x_frac >= (1.0 - self.lateral_zone)
        )

        # Solidité de la plage : un mufle est plein, deux sabots collés le sont
        # moins. On mesure le taux de remplissage de la sous-bande concernée.
        half = max(run_len // 2, 1)
        c0 = max(0, run_center - half)
        c1 = min(w, run_center + half + 1)
        sub = band[:, c0:c1]
        solidity = float(sub.sum()) / max(sub.size, 1)

        # ── Décision ────────────────────────────────────────
        if max_run_frac >= self.strong_run:
            conf = min(1.0, 0.70 + (max_run_frac - self.strong_run) * 2.0)
            if is_lateral:
                conf = min(1.0, conf + 0.15)
            return HeadState(True, conf, max_run_frac, head_x_frac,
                             f"mufle large au sol ({max_run_frac:.2f})")

        if max_run_frac >= self.run_threshold and is_lateral and solidity >= 0.35:
            span = max(self.strong_run - self.run_threshold, 1e-6)
            conf = 0.45 + 0.25 * (max_run_frac - self.run_threshold) / span
            return HeadState(True, min(conf, 0.95), max_run_frac, head_x_frac,
                             f"plage laterale au sol ({max_run_frac:.2f})")

        return HeadState(False, 0.0, max_run_frac, head_x_frac,
                         f"sabots seuls ({max_run_frac:.2f})")


# ─────────────────────────────────────────────────────────────
#  Persistance temporelle — "est-ce que ça bouge / ça dure ?"
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
#  Occlusion entre bovins
# ─────────────────────────────────────────────────────────────
def overlap_fractions(boxes) -> np.ndarray:
    """Pour chaque boîte, la fraction de SA surface recouverte par une autre.

    On mesure `intersection / aire_de_la_boîte`, et non l'IoU : ce qui compte
    ici n'est pas la ressemblance entre deux boîtes mais la part de l'animal
    qui est masquée. Un petit veau entièrement caché derrière une vache adulte
    a une IoU faible mais une fraction de recouvrement proche de 1.

    Sert à deux choses :
      - ne pas mettre à jour l'empreinte de ré-identification à partir d'une
        imagette polluée par un congénère (dérive de l'empreinte stockée) ;
      - ne pas conclure sur la posture quand la silhouette est tronquée.
    """
    arr = np.asarray(boxes, dtype=np.float32)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    if n == 1:
        return out

    areas = np.maximum(arr[:, 2] - arr[:, 0], 0) * np.maximum(arr[:, 3] - arr[:, 1], 0)
    for i in range(n):
        if areas[i] <= 0:
            continue
        x1 = np.maximum(arr[i, 0], arr[:, 0])
        y1 = np.maximum(arr[i, 1], arr[:, 1])
        x2 = np.minimum(arr[i, 2], arr[:, 2])
        y2 = np.minimum(arr[i, 3], arr[:, 3])
        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        inter[i] = 0.0  # ne pas se comparer à soi-même
        out[i] = float(inter.max()) / float(areas[i])
    return out


@dataclass
class HeadMotionTracker:
    """Suit la posture de tête d'un track sur une fenêtre glissante.

    Une vache qui broute garde la tête basse PLUSIEURS SECONDES et balaye
    latéralement (le mufle se déplace le long du sol). Un faux positif d'une
    frame isolée ne persiste pas. On exige donc `min_ratio` de frames en
    tête-basse sur la fenêtre avant de conclure.

    `sweep()` expose l'amplitude du balayage latéral du mufle — signature
    caractéristique du broutage, exploitable plus tard pour distinguer
    "broute activement" de "tête basse immobile" (reniflage, boisson).
    """
    window: int = 12
    min_ratio: float = 0.5
    _down_hist: dict[int, deque] = field(default_factory=dict)
    _x_hist: dict[int, deque] = field(default_factory=dict)

    def _buf(self, store: dict[int, deque], tid: int) -> deque:
        buf = store.get(tid)
        if buf is None or buf.maxlen != self.window:
            buf = deque(maxlen=self.window)
            store[tid] = buf
        return buf

    def update(self, tid: int, state: HeadState) -> bool:
        """Enregistre l'état courant, retourne le verdict lissé pour ce track."""
        tid = int(tid)
        down_buf = self._buf(self._down_hist, tid)
        down_buf.append(bool(state.head_down))
        if state.head_x_frac is not None:
            self._buf(self._x_hist, tid).append(float(state.head_x_frac))
        return self.is_grazing(tid)

    def is_grazing(self, tid: int) -> bool:
        buf = self._down_hist.get(int(tid))
        if not buf:
            return False
        return (sum(buf) / len(buf)) >= self.min_ratio

    def sweep(self, tid: int) -> float:
        """Amplitude du balayage latéral du mufle (écart-type, 0 si inconnu)."""
        buf = self._x_hist.get(int(tid))
        if not buf or len(buf) < 3:
            return 0.0
        return float(np.std(np.asarray(buf, dtype=np.float32)))

    def forget(self, tid: int) -> None:
        tid = int(tid)
        self._down_hist.pop(tid, None)
        self._x_hist.pop(tid, None)

    def prune(self, active_tids: set[int]) -> None:
        """Libère les tracks disparus."""
        for tid in list(self._down_hist.keys()):
            if tid not in active_tids:
                self.forget(tid)
