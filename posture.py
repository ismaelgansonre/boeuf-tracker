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

COUCHÉ : LE MÊME SIGNAL, POUSSÉ PLUS LOIN
-----------------------------------------
Une vache COUCHÉE pose tout son corps au sol. La bande basse n'est alors ni
"4 sabots fins" ni "sabots + mufle" : c'est un bloc quasi plein sur toute la
largeur de la boîte.

    debout   → max_run_frac ~0.05-0.10 , ground_fill ~0.15-0.30
    pâture   → max_run_frac ~0.15-0.35 , ground_fill ~0.30-0.45
    couché   → max_run_frac ~0.60-0.95 , ground_fill ~0.60-0.95

C'est pour cela qu'un bovin couché était étiqueté "pâture" : sa plage au sol
dépasse largement le seuil du mufle, `head_down` passait à True et l'analyseur
de comportement forçait "pâture". On lève désormais un drapeau `lying`
DISTINCT et prioritaire, et `head_down` est neutralisé quand `lying` est vrai.

GARDE-FOU INDISPENSABLE : LA TRONCATURE
---------------------------------------
Un bovin dont la boîte est coupée par le bord bas de l'image (gros plan) a lui
aussi une bande basse pleine — sans être couché : on ne voit simplement pas
ses pattes. `analyze(..., truncated_bottom=True)` interdit alors de conclure
"couché". Sans ce garde-fou, toute vache filmée de près est déclarée couchée.
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
    """Posture déduite d'un masque de segmentation.

    `lying` et `ground_fill` sont ajoutés en fin de dataclass avec valeurs par
    défaut : les constructions positionnelles existantes (5 arguments) restent
    valides.
    """
    head_down: bool
    confidence: float          # 0..1
    max_run_frac: float        # largeur de la plus longue plage au sol / largeur bbox
    head_x_frac: float | None  # position horizontale du mufle (0=gauche, 1=droite)
    reason: str                # explication courte (debug / UI)
    lying: bool = False        # corps entier posé au sol
    ground_fill: float = 0.0   # taux de remplissage de la bande basse (0..1)

    @classmethod
    def unknown(cls, reason: str = "masque inexploitable") -> "HeadState":
        return cls(False, 0.0, 0.0, None, reason)


# Alias explicite : la classe décrit désormais la posture complète (tête + corps).
PostureState = HeadState


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
    lying_run        : max_run_frac au-delà duquel la bande basse n'est plus un
                       mufle mais un flanc posé au sol.
    lying_fill       : remplissage minimal de la bande basse pour "couché".
                       Un mufle laisse les entre-pattes vides (~0.40) ; un corps
                       couché remplit la bande (~0.70).
    """
    band_frac: float = 0.18
    # Seuils releves (0.13/0.22 -> 0.16/0.28) : sur le terrain, un sabot vu
    # de biais ou une ombre depassait 0.13 et etiquetait "pature" des bovins
    # simplement immobiles. On prefere rater une frame de pature (le lissage
    # temporel la rattrape) que sur-declarer.
    run_threshold: float = 0.16
    strong_run: float = 0.28
    lateral_zone: float = 0.40
    min_mask_px: int = 200
    # 0.55/0.50 -> 0.62/0.58 : un veau broutant vu de trois-quarts remplissait
    # la bande basse au-dela de 0.55 et se faisait declarer couche.
    lying_run: float = 0.62
    lying_fill: float = 0.58

    def analyze(
        self,
        mask_roi: np.ndarray,
        truncated_bottom: bool = False,
    ) -> HeadState:
        """mask_roi : masque binaire (bool/uint8) recadré sur la bbox du bovin.

        truncated_bottom : la boîte touche le bord bas de l'image (ou est
        masquée en bas par un obstacle). Les pattes ne sont alors pas visibles
        et la bande basse est pleine même debout → on s'interdit de conclure
        "couché".
        """
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
        # Taux de remplissage de la bande : discrimine un flanc couché (plein)
        # d'un mufle posé entre des pattes (creux entre les sabots).
        ground_fill = float(band.sum()) / float(band.size)

        if run_center < 0:
            return HeadState.unknown("aucune plage au sol")

        head_x_frac = run_center / w

        # ── Couché : le corps entier repose au sol ───────────
        # Prioritaire sur la tête basse — un bovin couché présente une plage
        # au sol bien plus large qu'un mufle, ce qui déclenchait à tort
        # "pâture". `head_down` est explicitement remis à False.
        # Un bovin couche est TOUJOURS plus large que haut. Une silhouette
        # portrait (h > w : animal de face, ou veau tete baissee) ne peut pas
        # etre un corps au sol, quel que soit le remplissage de la bande.
        if max_run_frac >= self.lying_run and ground_fill >= self.lying_fill \
                and w > h:
            if truncated_bottom:
                return HeadState(
                    False, 0.0, max_run_frac, head_x_frac,
                    f"bas de silhouette tronque ({max_run_frac:.2f})",
                    lying=False, ground_fill=ground_fill,
                )
            conf = min(1.0, 0.60 + (max_run_frac - self.lying_run) * 1.2)
            return HeadState(
                False, conf, max_run_frac, head_x_frac,
                f"corps au sol ({max_run_frac:.2f}, rempli {ground_fill:.2f})",
                lying=True, ground_fill=ground_fill,
            )
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
                             f"mufle large au sol ({max_run_frac:.2f})",
                             lying=False, ground_fill=ground_fill)

        if max_run_frac >= self.run_threshold and is_lateral and solidity >= 0.35:
            span = max(self.strong_run - self.run_threshold, 1e-6)
            conf = 0.45 + 0.25 * (max_run_frac - self.run_threshold) / span
            return HeadState(True, min(conf, 0.95), max_run_frac, head_x_frac,
                             f"plage laterale au sol ({max_run_frac:.2f})",
                             lying=False, ground_fill=ground_fill)

        return HeadState(False, 0.0, max_run_frac, head_x_frac,
                         f"sabots seuls ({max_run_frac:.2f})",
                         lying=False, ground_fill=ground_fill)


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
    # 0.5 -> 0.65 : il faut une nette majorite de frames tete-basse sur la
    # fenetre avant de conclure "pature" — une vache qui broute vraiment tient
    # la tete au sol quasi en continu, un artefact non.
    min_ratio: float = 0.65
    lying_min_ratio: float = 0.6     # seuil de SORTIE (hystérésis basse)
    lying_enter_ratio: float = 0.80  # seuil d'ENTRÉE (hystérésis haute)
    lying_min_frames: int = 6
    _down_hist: dict[int, deque] = field(default_factory=dict)
    _x_hist: dict[int, deque] = field(default_factory=dict)
    _lying_hist: dict[int, deque] = field(default_factory=dict)
    _lying_state: dict[int, bool] = field(default_factory=dict)

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
        self._buf(self._lying_hist, tid).append(bool(state.lying))
        if state.head_x_frac is not None:
            self._buf(self._x_hist, tid).append(float(state.head_x_frac))
        return self.is_grazing(tid)

    def is_grazing(self, tid: int) -> bool:
        buf = self._down_hist.get(int(tid))
        if not buf:
            return False
        # Un bovin couché n'est pas en train de paître : le corps au sol
        # produit une plage large qui imiterait un mufle.
        if self.is_lying(tid):
            return False
        return (sum(buf) / len(buf)) >= self.min_ratio

    def is_lying(self, tid: int) -> bool:
        """Verdict "couché", à hystérésis (déclencheur de Schmitt).

        Trois exigences, plus strictes que pour la pâture :

        - `lying_min_frames` observations avant de conclure quoi que ce soit.
          Un bovin ne se couche pas en deux frames ; sans ce minimum, les
          toutes premières frames d'une piste (ratio 1.0 sur 1 échantillon)
          suffiraient à le déclarer couché.

        - DEUX seuils, pas un. Il faut `lying_enter_ratio` (0.80) pour passer
          debout → couché, mais il suffit de repasser sous `lying_min_ratio`
          (0.60) pour ressortir. Avec un seuil unique, une séquence dont le
          taux oscille autour du seuil bascule à chaque frame : mesuré sur
          samples/IMG_3544, cela produisait 159 transitions "se couche" en
          60 frames — un journal d'événements inexploitable. L'écart entre les
          deux seuils est la marge de bruit.

        - une majorité sur la fenêtre : une silhouette momentanément fusionnée
          avec un congénère produit une bande basse pleine, mais pas durablement.
        """
        tid = int(tid)
        buf = self._lying_hist.get(tid)
        if not buf or len(buf) < self.lying_min_frames:
            return False
        ratio = sum(buf) / len(buf)
        current = self._lying_state.get(tid, False)
        threshold = self.lying_min_ratio if current else self.lying_enter_ratio
        current = ratio >= threshold
        self._lying_state[tid] = current
        return current

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
        self._lying_hist.pop(tid, None)
        self._lying_state.pop(tid, None)

    def prune(self, active_tids: set[int]) -> None:
        """Libère les tracks disparus."""
        for tid in list(self._down_hist.keys()):
            if tid not in active_tids:
                self.forget(tid)


# Alias : le tracker suit désormais tête basse ET corps au sol.
PostureTracker = HeadMotionTracker
MaskPostureAnalyzer = MaskHeadAnalyzer
