"""
detector.py
-----------
Wrapper autour de YOLOv11 (Ultralytics) avec segmentation d'instance.
task=segment produit des masques par pixel qui suivent la silhouette exacte
du bovin, au lieu de simples rectangles.

FP16 (half precision) activé par défaut sur CUDA. On convertit les poids
du modèle via `model.model.half()` une seule fois (la nouvelle API
Ultralytics a déprécié le paramètre `half=` de predict/track).

ET
-------
CattleDetectorMLX: YOLO26 sur MLX (Apple Metal GPU) avec tracking ByteTrack natif.
YOLO26 MLX est ~2.6× plus rapide que YOLO11s sur PyTorch MPS pour Apple Silicon M1/M2/M3/M4.
"""
import numpy as np
import torch
from ultralytics import YOLO

from console import info, ok, warn

# ─────────────────────────────────────────────────────────────────────────
#  Classes COCO acceptées comme "bovin"
# ─────────────────────────────────────────────────────────────────────────
# COCO ne contient qu'une classe 'cow' (19), entraînée sur des photos de
# prairie. Sur des images de ferme — troupeau serré, animaux de dos, sujets
# lointains, veaux — le réseau hésite entre 'cow', 'horse' (17) et 'sheep'
# (18) : le score se répartit entre les trois et aucun ne passe le seuil.
# Accepter les trois récupère ces animaux au lieu de les perdre.
#
# À désactiver (`extra_classes=False`) si des chevaux ou des moutons partagent
# réellement l'enclos, sans quoi ils seront comptés comme bovins.
COW_CLASS_ID = 19
HORSE_CLASS_ID = 17
SHEEP_CLASS_ID = 18
CATTLE_CLASS_IDS = (COW_CLASS_ID, HORSE_CLASS_ID, SHEEP_CLASS_ID)


def dedup_boxes(
    xyxy: np.ndarray,
    conf: np.ndarray,
    contain_frac: float = 0.75,
    iou_thr: float = 0.75,
) -> np.ndarray:
    """Indices à conserver après suppression des boîtes redondantes.

    POURQUOI, ALORS QUE YOLO26 EST end2end (SANS NMS)
    -------------------------------------------------
    YOLO26 apprend à ne sortir qu'une boîte par objet, et se passe donc de NMS.
    Cet apprentissage tient à la résolution d'entraînement ; en montant `imgsz`
    (ce qu'on fait pour récupérer les bovins lointains), un même animal
    ressort parfois 2-3 fois : une boîte sur le corps, une sur l'avant-train.
    Ces doublons créeraient autant de fausses identités en base.
    """
    n = len(xyxy)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    areas = (np.maximum(xyxy[:, 2] - xyxy[:, 0], 0)
             * np.maximum(xyxy[:, 3] - xyxy[:, 1], 0))
    keep: list[int] = []
    for i in np.argsort(-np.asarray(conf)):  # du plus sûr au moins sûr
        i = int(i)
        redundant = False
        for j in keep:
            x1 = max(xyxy[i, 0], xyxy[j, 0])
            y1 = max(xyxy[i, 1], xyxy[j, 1])
            x2 = min(xyxy[i, 2], xyxy[j, 2])
            y2 = min(xyxy[i, 3], xyxy[j, 3])
            inter = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
            if inter <= 0.0:
                continue
            # Boîte essentiellement contenue dans une boîte déjà retenue :
            # l'IoU ne le voit pas (aires très différentes), le recouvrement si.
            if inter / max(float(areas[i]), 1e-6) > contain_frac:
                redundant = True
                break
            union = float(areas[i]) + float(areas[j]) - inter
            if inter / max(union, 1e-6) > iou_thr:
                redundant = True
                break
        if not redundant:
            keep.append(i)
    return np.array(sorted(keep), dtype=np.int64)


class CattleDetector:
    COW_CLASS_ID = COW_CLASS_ID  # COCO: 'cow'

    def __init__(
        self,
        model_name: str = "yolo11n-seg.pt",
        device: str | None = None,
        half: bool | None = None,
        extra_classes: bool = True,
    ):
        self.classes = list(CATTLE_CLASS_IDS) if extra_classes else [COW_CLASS_ID]
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        # FP16 uniquement sur CUDA (les CPU ne supportent pas demi-précision
        # efficacement et ça plante)
        if half is None:
            half = device.startswith("cuda") or device == "mps"
        self.half = half

        info(f"[YOLO] Chargement de {model_name} sur {device} (FP16={half})...")
        self.model = YOLO(model_name)

        # Conversion FP16 UNE SEULE FOIS sur les poids (evite le warning
        # "'half' is deprecated, use 'quantize' instead" d'Ultralytics 8.4+).
        if self.half and self.device.startswith("cuda"):
            try:
                self.model.model.half()
            except Exception as e:
                warn(f"[YOLO] FP16 impossible ({e}), fallback FP32")
                self.half = False

        # Warmup sur device + dtype cible
        try:
            dummy = [[[0] * 3] * 64 * 64]
            with torch.no_grad():
                self.model.predict(
                    source=dummy,
                    device=self.device,
                    verbose=False,
                )
        except Exception:
            pass
        ok(f"[YOLO] Modele {model_name} pret ({device}, FP16={self.half})")

    def detect(self, frame, persist: bool = True, conf: float = 0.25, imgsz: int = 960):
        """Detecte + track + segmente les bovins. Retourne results[0].

        imgsz: taille d'inference. 640 = rapide, 1280 = +precis mais +lent.
        Note: on ne passe plus `half=` (deprecie) -- le modele est deja en FP16
        si self.half=True (cf. __init__).

        `iou=0.7` : NMS plus permissif que le defaut. Dans un troupeau serre,
        deux bovins cote a cote ont des boites qui se recouvrent largement ;
        un seuil bas en supprimerait un.
        """
        return self.model.track(
            frame,
            classes=self.classes,
            persist=persist,
            device=self.device,
            verbose=False,
            conf=conf,
            iou=0.7,
            max_det=100,
            tracker="bytetrack.yaml",
            imgsz=imgsz,
        )[0]


import threading


class CattleDetectorMLX:
    """
    YOLO26 sur MLX (Apple Metal GPU) avec ByteTrack natif.
    ~2.6× plus rapide que YOLO11s sur PyTorch MPS pour Apple Silicon.

    Unterschiede gegenüber CattleDetector:
    - Utilise YOLO26 MLX au lieu de YOLO11 Ultralytics
    - MLX utilise Metal GPU directement (pas d'émulation MPS)
    - Le tracking ByteTrack est integre dans YOLO26 MLX
    - device est toujours "mlx" (MLX gere le hardware automatiquement)

    Note: yolo26mlx a un bug de thread-safety avec Metal (crash SIGABRT
    si deux appels sont en cours simultanement). On sérialise les appels
    avec un lock pour éviter ce crash.
    """
    COW_CLASS_ID = COW_CLASS_ID  # COCO: 'cow' (same as Ultralytics)

    def __init__(
        self,
        model_name: str = "yolo26s-seg.safetensors",
        device: str = "mlx",
        extra_classes: bool = True,
    ):
        self.device = device
        self.half = True  # MLX utilise FP16 nativement
        self.classes = np.array(
            CATTLE_CLASS_IDS if extra_classes else (COW_CLASS_ID,)
        )
        self._mlx_lock = threading.Lock()  # instance lock (thread-safety Metal)
        self._iou_tracker = _SimpleIoUTracker(iou_threshold=0.25, max_lost=45)

        info(f"[YOLO26-MLX] Chargement de {model_name} sur MLX (Metal GPU)...")

        # Auto-detect: use safetensors if available, else convert from .pt
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Si le modele passe est un chemin absolu ou un nom de fichier simple
        # qui existe dans le repertoire du projet, utiliser ce chemin.
        # Sinon, utiliser le modele tel quel (YOLO le telechargera si necessaire).
        if os.path.isabs(model_name):
            model_path = model_name
        elif os.path.exists(model_name):
            # Fichier local dans le cwd
            model_path = model_name
        else:
            # Construire le chemin dans le repertoire du projet
            local_safetensors = os.path.join(base_dir, "yolo26s-seg.safetensors")
            local_pt = os.path.join(base_dir, "yolo26s-seg.pt")

            if os.path.exists(local_safetensors):
                model_path = local_safetensors
            elif os.path.exists(local_pt):
                model_path = local_pt
            else:
                # Modele par defaut ou telechargement
                model_path = model_name

        info(f"[YOLO26-MLX] Modele final: {model_path}")
        # IMPORTANT: utiliser yolo26mlx.YOLO, pas ultralytics.YOLO
        from yolo26mlx import YOLO as MLX_YOLO
        self.model = MLX_YOLO(model_path)
        ok(f"[YOLO26-MLX] Modele pret (Metal GPU)")

    def detect(self, frame, persist: bool = True, conf: float = 0.25, imgsz: int = 960):
        """Detecte + segmente les bovins. Retourne un objet compatible avec
        le format Ultralytics (boxes.xyxy, boxes.id, boxes.conf, masks).

        _detect_inner fait déjà le filtrage 'cow' (cls == 19) + le tracking
        IoU et retourne un _MLXResult prêt à consommer. On ne re-filtre PAS
        ici : _MLXBoxes.cls est un placeholder (np.zeros) qui n'est pas la
        vraie classe — un double filtrage ici éliminerait toutes les
        détections (0 == 19 → False).

        Le lock sérialise les appels pour éviter un crash Metal (SIGABRT)
        quand deux appels MLX sont simultanés (Flask threaded mode).
        """
        with self._mlx_lock:
            return self._detect_inner(frame, persist, conf, imgsz)

    def _detect_inner(self, frame, persist, conf, imgsz):
        """Inner detect with simple retry (no model reinit to avoid crash).

        On utilise predict() au lieu de track() car yolo26mlx a un bug dans
        TrackerManager.update() qui perd les masques (il recree un Results
        sans masks). predict() retourne bien boxes + masks, et notre IoU
        tracker ci-dessous maintient les IDs entre frames.

        Retourne un _MLXResult directement (boxes filtrees bovins + masks)
        pour eviter tout probleme d'attributs sur l'objet Boxes yolo26mlx.
        """
        for attempt in range(2):
            try:
                pred = self.model.predict(
                    frame,
                    conf=conf,
                    imgsz=imgsz,
                )[0]

                boxes = pred.boxes
                if boxes is None or len(boxes.xyxy) == 0:
                    return _empty_result_mlx()

                xyxy_all = np.array(boxes.xyxy)
                cls_all = np.array(boxes.cls) if boxes.cls is not None else None
                conf_all = np.array(boxes.conf) if boxes.conf is not None else None

                # Filtrer sur les classes assimilees a du betail (cf.
                # CATTLE_CLASS_IDS : cow, et par defaut horse/sheep que COCO
                # confond avec des bovins sur des vues de troupeau).
                cow_mask = (
                    np.isin(cls_all, self.classes)
                    if cls_all is not None
                    else np.ones(len(xyxy_all), dtype=bool)
                )
                if not cow_mask.any():
                    return _empty_result_mlx()

                xyxy_cows = xyxy_all[cow_mask]
                conf_cows = conf_all[cow_mask] if conf_all is not None else np.ones(len(xyxy_cows))

                masks_data = None
                if pred.masks is not None and pred.masks.data is not None:
                    masks_all = np.array(pred.masks.data)
                    masks_data = masks_all[cow_mask]

                # Doublons imbriques : frequents des qu'on monte imgsz.
                keep = dedup_boxes(xyxy_cows, conf_cows)
                if len(keep) < len(xyxy_cows):
                    xyxy_cows = xyxy_cows[keep]
                    conf_cows = conf_cows[keep]
                    if masks_data is not None:
                        masks_data = masks_data[keep]

                # Appliquer le tracker IoU sur les boxes de bovins uniquement
                ids_cows = self._iou_tracker.update(xyxy_cows)

                return _MLXResult(xyxy_cows, ids_cows, conf_cows, masks_data)
            except Exception as e:
                if attempt < 1:
                    warn(f"[YOLO26-MLX] Retry {attempt+1} after error: {e}")
                else:
                    warn(f"[YOLO26-MLX] All retries failed: {e}")
                    return _empty_result_mlx()


class _MLXResult:
    """Result object compatible with Ultralytics Result for cows only."""

    def __init__(self, xyxy, ids, confs, masks_data):
        self.xyxy = xyxy
        self.id = ids
        self.conf = confs
        self.masks_data = masks_data

    @property
    def boxes(self):
        return _MLXBoxes(self.xyxy, self.id, self.conf)

    @property
    def masks(self):
        if self.masks_data is None:
            return None
        return _MLXMasks(self.masks_data)


class _NumpyWrapper:
    """Wrapper around numpy array that provides .cpu() for compatibility
    with processor.py which expects torch tensors.

    The chain .cpu().numpy() and .int().cpu().numpy() from processor.py
    is supported by having .cpu() return an object that itself has .numpy()
    and .int() methods.
    """

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        """Returns a numpy-array-backed object that supports .numpy() and .int()."""
        return _CpuView(self._arr)

    def int(self):
        """Returns int version (for track IDs)."""
        return _CpuView(self._arr.astype(np.int32))

    def numpy(self):
        """Returns the numpy array (direct call, no .cpu() first)."""
        return self._arr

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, key):
        return self._arr[key]

    def __array__(self):
        return self._arr


class _CpuView:
    """Object returned by _NumpyWrapper.cpu() and .int(). Supports .numpy()."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def numpy(self):
        return self._arr

    def int(self):
        return self._arr.astype(np.int32)

    def cpu(self):
        return self  # already on "cpu"

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, key):
        return self._arr[key]

    def __array__(self):
        return self._arr


class _MLXBoxes:
    """Minimal boxes implementation compatible with processor.py usage."""

    def __init__(self, xyxy, ids, confs):
        # Wrap in _NumpyWrapper so .cpu().numpy() works (processor.py pattern)
        self.xyxy = _NumpyWrapper(xyxy) if xyxy is not None else None
        self.id = _NumpyWrapper(ids) if ids is not None else None
        self.conf = _NumpyWrapper(confs) if confs is not None else None
        self.cls = np.zeros(len(xyxy))  # dummy, not used in processor

    def __len__(self):
        return len(self.xyxy._arr) if self.xyxy is not None else 0


class _MLXMasks:
    """Minimal masks implementation compatible with processor.py usage."""

    def __init__(self, data):
        # data shape: (N, H, W) where H,W are mask dimensions
        self.data = _NumpyWrapper(data)  # wrap so .cpu().numpy() works

    def __len__(self):
        return len(self.data._arr)


def _empty_result_mlx():
    """Return an empty result when no cows detected."""
    empty_boxes = _MLXBoxes(
        np.empty((0, 4)), np.array([]), np.array([])
    )
    return type('EmptyResult', (), {
        'boxes': empty_boxes,
        'masks': None,
    })()


try:
    from scipy.optimize import linear_sum_assignment as _hungarian
    _SCIPY_OK = True
except Exception:  # pragma: no cover - repli si scipy absent
    _SCIPY_OK = False


def _iou_matrix(dets: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """Matrice IoU (n_det, n_track), vectorisee."""
    if len(dets) == 0 or len(tracks) == 0:
        return np.zeros((len(dets), len(tracks)), dtype=np.float32)
    d = dets[:, None, :]
    t = tracks[None, :, :]
    x1 = np.maximum(d[..., 0], t[..., 0])
    y1 = np.maximum(d[..., 1], t[..., 1])
    x2 = np.minimum(d[..., 2], t[..., 2])
    y2 = np.minimum(d[..., 3], t[..., 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_d = np.clip(d[..., 2] - d[..., 0], 0, None) * np.clip(d[..., 3] - d[..., 1], 0, None)
    area_t = np.clip(t[..., 2] - t[..., 0], 0, None) * np.clip(t[..., 3] - t[..., 1], 0, None)
    union = area_d + area_t - inter
    return (inter / np.maximum(union, 1e-6)).astype(np.float32)


class _SimpleIoUTracker:
    """Tracker par association IoU pour YOLO26 MLX.

    yolo26mlx.TrackerManager.update() perd les masques (il recree un Results
    sans masks quand il applique ByteTrack). Comme on ne peut pas utiliser
    track() pour avoir les masques, on utilise predict() et on maintient
    nos propres IDs.

    ASSOCIATION GLOBALE, PAS GLOUTONNE
    ----------------------------------
    L'association se fait par affectation optimale (Hongrois) sur toute la
    matrice IoU, pas boite par boite. Dans un troupeau serre, l'appariement
    glouton attribue la piste au PREMIER bovin qui la recouvre suffisamment,
    même si un autre la recouvre bien mieux : les identites s'echangent entre
    voisins, et le nom suit l'echange. L'affectation globale minimise le cout
    total et supprime cette classe d'erreurs.

    DEUXIEME PASSE, PLUS PERMISSIVE
    -------------------------------
    Les pistes non appariees au premier tour (bovin brievement masque, ou qui
    s'est deplace vite) sont re-tentees avec un seuil IoU abaisse, en ne
    considerant que les pistes recemment perdues. Sans cela, une occlusion
    d'une seconde cree un nouvel identifiant — donc un nouveau nom en base.

    Une piste perdue est conservee `max_lost` frames : elle peut reprendre son
    identifiant en reapparaissant.
    """

    def __init__(self, iou_threshold: float = 0.25, max_lost: int = 45):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self._next_id = 1
        # Liste de dicts {box, id, lost}
        self._tracks: list[dict] = []

    def _iou(self, box_a, box_b):
        """Calcule l'IoU entre deux boxes (format xyxy)."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _assign(self, iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
        """Apparie detections et pistes. Retourne [(i_det, j_track), ...]."""
        if iou.size == 0:
            return []
        pairs: list[tuple[int, int]] = []
        if _SCIPY_OK:
            rows, cols = _hungarian(-iou)  # maximise l'IoU total
            for i, j in zip(rows, cols):
                if iou[i, j] >= threshold:
                    pairs.append((int(i), int(j)))
            return pairs
        # Repli glouton par IoU decroissante (scipy absent)
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        used_d, used_t = set(), set()
        for i, j in order:
            i, j = int(i), int(j)
            if iou[i, j] < threshold:
                break
            if i in used_d or j in used_t:
                continue
            used_d.add(i)
            used_t.add(j)
            pairs.append((i, j))
        return pairs

    def update(self, xyxy_boxes):
        """Met a jour le tracker avec les nouvelles boxes. Retourne les IDs."""
        dets = np.asarray(xyxy_boxes, dtype=np.float32).reshape(-1, 4)

        # Incrementer l'age des tracks existants
        for track in self._tracks:
            track["lost"] += 1

        ids = np.zeros(len(dets), dtype=np.int64)
        if len(self._tracks) and len(dets):
            track_boxes = np.array([t["box"] for t in self._tracks], dtype=np.float32)
            iou = _iou_matrix(dets, track_boxes)

            pairs = self._assign(iou, self.iou_threshold)
            matched_d = {i for i, _ in pairs}
            matched_t = {j for _, j in pairs}

            # Seconde passe : seuil abaisse, uniquement sur les pistes
            # recemment perdues (un bovin qui reapparait apres occlusion).
            free_d = [i for i in range(len(dets)) if i not in matched_d]
            free_t = [j for j in range(len(self._tracks))
                      if j not in matched_t and self._tracks[j]["lost"] > 1]
            if free_d and free_t:
                sub = iou[np.ix_(free_d, free_t)]
                pairs += [(free_d[i], free_t[j])
                          for i, j in self._assign(sub, self.iou_threshold * 0.5)]

            # Applique les appariements retenus
            for i, j in pairs:
                track = self._tracks[j]
                track["box"] = dets[i]
                track["lost"] = 0
                ids[i] = track["id"]

        # Detections non appariees -> nouvelles pistes
        for i in range(len(dets)):
            if ids[i] == 0:
                new_id = self._next_id
                self._next_id += 1
                self._tracks.append({"box": dets[i], "id": new_id, "lost": 0})
                ids[i] = new_id

        # Nettoyer les tracks perdues depuis trop longtemps
        self._tracks = [t for t in self._tracks if t["lost"] <= self.max_lost]

        return ids
