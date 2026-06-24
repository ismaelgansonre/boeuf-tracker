"""
detector.py
-----------
Wrapper autour de YOLOv11 (Ultralytics) avec segmentation d'instance.
task=segment produit des masques par pixel qui suivent la silhouette exacte
du bovin, au lieu de simples rectangles.

FP16 (half precision) activé par défaut sur CUDA — double la vitesse sur
les GPU NVIDIA (Tensor Cores).
"""
import torch
from ultralytics import YOLO


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

        print(f"[YOLO] Chargement de {model_name} sur {device} (FP16={half})...")
        self.model = YOLO(model_name)
        # Warmup sur device + dtype cible
        try:
            self.model.predict(
                source=[[[0] * 3] * 64 * 64],  # dummy frame 64x64x3
                device=self.device,
                half=self.half,
                verbose=False,
            )
        except Exception:
            pass
        print(f"[YOLO] Modèle {model_name} prêt")

    def detect(self, frame, persist: bool = True, conf: float = 0.4):
        """Détecte + track + segmente les bovins. Retourne results[0]."""
        return self.model.track(
            frame,
            classes=[self.COW_CLASS_ID],
            persist=persist,
            device=self.device,
            verbose=False,
            conf=conf,
            tracker="bytetrack.yaml",
            half=self.half,
        )[0]