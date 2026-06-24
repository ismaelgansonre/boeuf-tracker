"""
reid.py
-------
Extraction d'embeddings visuels via DINOv2 (Meta).
DINOv2 est excellent pour capturer les motifs de pelage sans entraînement.
Modèle par défaut: facebook/dinov2-small (léger, ~250 MB, GPU-friendly).
"""
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor


class CattleReID:
    def __init__(self, model_name: str = "facebook/dinov2-small", device: str = "cpu"):
        self.device = device
        print(f"[DINOv2] Chargement de {model_name} sur {device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        # Pré-chauffage pour ne pas pénaliser la première frame
        with torch.no_grad():
            dummy_pil = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            dummy_in = self.processor(images=dummy_pil, return_tensors="pt").to(self.device)
            _ = self.model(**dummy_in)
        print(f"[DINOv2] Modèle prêt")

    @torch.no_grad()
    def get_embedding(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Extrait un vecteur d'embedding normalisé (L2) à partir d'un crop BGR."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 16 or w < 16:
            return None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        # Mean pooling sur les patch tokens
        emb = out.last_hidden_state.mean(dim=1).flatten().cpu().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb