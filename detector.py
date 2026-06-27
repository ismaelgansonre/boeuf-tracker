"""
detector.py
-----------
Wrapper autour de YOLOv11 (Ultralytics) avec segmentation d'instance.
task=segment produit des masques par pixel qui suivent la silhouette exacte
du bovin, au lieu de simples rectangles.

FP16 (half precision) activé par défaut sur CUDA. On convertit les poids
du modèle via `model.model.half()` une seule fois (la nouvelle API
Ultralytics a déprécié le paramètre `half=` de predict/track).
"""
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
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # FP16 uniquement sur CUDA (les CPU ne supportent pas demi-précision
        # efficacement et ça plante)
        if half is None:
            half = device.startswith("cuda")
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
