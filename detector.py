"""
detector.py
-----------
Wrapper autour de YOLOv11 (Ultralytics) pour la détection + tracking intra-vidéo.
Utilise la classe COCO 19 ("cow"). Pour des bœufs spécifiques, fine-tunez sur
votre propre dataset et passez le chemin de votre modèle via --yolo-model.
"""
import torch
from ultralytics import YOLO


class CattleDetector:
    COW_CLASS_ID = 19  # COCO: 'cow'

    def __init__(self, model_name: str = "yolo11n.pt", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"[YOLO] Chargement de {model_name} sur {device}...")
        self.model = YOLO(model_name)
        print(f"[YOLO] Modèle {model_name} prêt (warmup au premier frame)")

    def detect(self, frame, persist: bool = True, conf: float = 0.4):
        """Détecte + track les bovins. Retourne results[0] (Results Ultralytics)."""
        return self.model.track(
            frame,
            classes=[self.COW_CLASS_ID],
            persist=persist,
            device=self.device,
            verbose=False,
            conf=conf,
            tracker="botsort.yaml",
        )[0]