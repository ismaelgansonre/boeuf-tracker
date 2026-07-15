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


class CattleDetector:
    COW_CLASS_ID = 19  # COCO: 'cow'

    def __init__(
        self,
        model_name: str = "yolo11n-seg.pt",
        device: str | None = None,
        half: bool | None = None,
    ):
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

    def detect(self, frame, persist: bool = True, conf: float = 0.4, imgsz: int = 640):
        """Detecte + track + segmente les bovins. Retourne results[0].

        imgsz: taille d'inference. 640 = rapide, 1280 = +precis mais +lent.
        Note: on ne passe plus `half=` (deprecie) -- le modele est deja en FP16
        si self.half=True (cf. __init__).
        """
        return self.model.track(
            frame,
            classes=[self.COW_CLASS_ID],
            persist=persist,
            device=self.device,
            verbose=False,
            conf=conf,
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
    COW_CLASS_ID = 19  # COCO: 'cow' (same as Ultralytics)

    def __init__(self, model_name: str = "yolo26s-seg.safetensors", device: str = "mlx"):
        self.device = device
        self.half = True  # MLX utilise FP16 nativement
        self._mlx_lock = threading.Lock()  # instance lock (thread-safety Metal)
        self._iou_tracker = _SimpleIoUTracker(iou_threshold=0.3, max_lost=30)

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

    def detect(self, frame, persist: bool = True, conf: float = 0.4, imgsz: int = 640):
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

                # Filtrer pour garder uniquement les 'cow' (cls == 19)
                cow_mask = cls_all == self.COW_CLASS_ID if cls_all is not None else np.ones(len(xyxy_all), dtype=bool)
                if not cow_mask.any():
                    return _empty_result_mlx()

                xyxy_cows = xyxy_all[cow_mask]
                conf_cows = conf_all[cow_mask] if conf_all is not None else np.ones(len(xyxy_cows))

                # Appliquer le tracker IoU sur les boxes de bovins uniquement
                ids_cows = self._iou_tracker.update(xyxy_cows)

                masks_data = None
                if pred.masks is not None and pred.masks.data is not None:
                    masks_all = np.array(pred.masks.data)
                    masks_data = masks_all[cow_mask]

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


class _SimpleIoUTracker:
    """Tracker IoU simple pour YOLO26 MLX.

    yolo26mlx.TrackerManager.update() perd les masques (il recree un Results
    sans masks quand il applique ByteTrack). Comme on ne peut pas utiliser
    track() pour avoir les masques, on utilise predict() et on maintient
    nos propres IDs par IoU matching.

    Strategie:
    - Pour chaque nouvelle detection, on cherche la box precedente avec
      l'IoU le plus eleve.
    - Si IoU > seuil, on conserve le meme ID.
    - Sinon, on assigne un nouvel ID.
    - Les tracks perdues depuis >max_lost frames sont retirees.

    Note: ce tracker est basique mais fonctionne pour des sequences video
    ou les bovins bougent peu entre frames consecutives. Pour du tracking
    plus robuste (occlusions, mouvements rapides), il faudrait implementer
    ByteTrack ou SORT/DeepSORT. Pour notre cas d'usage (vaches dans un
    champ), c'est suffisant.
    """

    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 30):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self._next_id = 1
        # Liste de tuples (xyxy, track_id, frames_since_last_seen)
        self._tracks = []

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

    def update(self, xyxy_boxes):
        """Met a jour le tracker avec les nouvelles boxes. Retourne les IDs."""
        # Incrementer l'age des tracks existants
        for track in self._tracks:
            track["lost"] += 1

        ids = []
        used_track_indices = set()

        for box in xyxy_boxes:
            best_iou = 0.0
            best_idx = -1

            for idx, track in enumerate(self._tracks):
                if idx in used_track_indices:
                    continue
                iou = self._iou(box, track["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= self.iou_threshold and best_idx >= 0:
                # Match: conserver l'ID existant
                track = self._tracks[best_idx]
                track["box"] = box
                track["lost"] = 0
                ids.append(track["id"])
                used_track_indices.add(best_idx)
            else:
                # Nouvelle detection
                new_id = self._next_id
                self._next_id += 1
                self._tracks.append({"box": box, "id": new_id, "lost": 0})
                ids.append(new_id)

        # Nettoyer les tracks perdues depuis trop longtemps
        self._tracks = [
            t for t in self._tracks if t["lost"] <= self.max_lost
        ]

        return np.array(ids, dtype=np.int64)
