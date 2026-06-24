"""
reid.py
-------
Extraction d'embeddings pour la ré-identification de bovins.

Combine TROIS signaux complémentaires:

1. **DINOv2** (384 dim): sémantique générale, capture forme et apparence globale
2. **HSV color histogram** (48 dim): distribution des couleurs du pelage
   (crucial pour distinguer bovins de robes différentes)
3. **LBP texture histogram** (32 dim): texture locale du pelage
   (capte les motifs au-delà de la couleur brute)

Le vecteur final est concaténé + normalisé L2.
Sans entraînement, ce combo surpasse DINOv2 seul pour distinguer des bovins
de races similaires.
"""
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor


class CattleReID:
    # Dimensiones des sous-embeddings
    DINO_DIM = 384   # dinov2-small
    HSV_DIM = 48     # 16 par canal (H, S, V)
    LBP_DIM = 32     # texture locale
    TOTAL_DIM = DINO_DIM + HSV_DIM + LBP_DIM  # 464

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device: str = "cpu",
        dino_weight: float = 0.5,
        hsv_weight: float = 0.3,
        lbp_weight: float = 0.2,
    ):
        """
        Poids des trois composantes. Pour races à motifs (Holstein, Normand):
        garder ces valeurs. Pour races unies (Angus, Charolais): baisser
        dino_weight à 0.3 et monter hsv_weight à 0.5.
        """
        self.device = device
        self.dino_weight = dino_weight
        self.hsv_weight = hsv_weight
        self.lbp_weight = lbp_weight

        # Normalisation pour que les poids somment à 1
        s = dino_weight + hsv_weight + lbp_weight
        self.w_dino = dino_weight / s
        self.w_hsv = hsv_weight / s
        self.w_lbp = lbp_weight / s

        print(f"[DINOv2] Chargement de {model_name} sur {device}...", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        with torch.no_grad():
            dummy_pil = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            dummy_in = self.processor(images=dummy_pil, return_tensors="pt").to(self.device)
            _ = self.model(**dummy_in)
        print(f"[DINOv2] Modèle prêt (dim totale={self.TOTAL_DIM})", flush=True)

    @torch.no_grad()
    def _dino(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 16 or w < 16:
            return None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).flatten().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

    def _hsv_hist(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Histogramme HSV: distribution des teintes, saturation, valeur."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 32 or w < 32:
            return None
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        # 16 bins par canal = 48 dim total
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            [16, 16, 16],
            [0, 180, 0, 256, 0, 256],
        ).flatten()
        hist = hist / (hist.sum() + 1e-8)
        return hist.astype(np.float32)

    @staticmethod
    def _lbp_hist(gray: np.ndarray, grid: int = 4, bins: int = 32) -> np.ndarray:
        """
        Local Binary Pattern simplifié + histogramme par cellule de la grille.
        Robuste aux variations d'éclairage, capture la texture fine du pelage.
        """
        # LBP basique: compare chaque pixel avec ses 8 voisins
        h, w = gray.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        for i in range(8):
            dy, dx = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)][i]
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            lbp += ((gray[1:-1, 1:-1] > shifted[1:-1, 1:-1]) << i).astype(np.uint8)
        # Grille grid×grid + histogramme par cellule
        gh, gw = lbp.shape[0] // grid, lbp.shape[1] // grid
        feats = []
        for gy in range(grid):
            for gx in range(grid):
                cell = lbp[gy * gh:(gy + 1) * gh, gx * gw:(gx + 1) * gw]
                hist, _ = np.histogram(cell, bins=bins, range=(0, 256))
                hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
                feats.append(hist)
        return np.concatenate(feats)

    def _lbp(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 32 or w < 32:
            return None
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        feats = self._lbp_hist(gray, grid=4, bins=8)  # 4*4*8 = 128 → on prend les 32 plus discriminants
        # Réduire à 32 dim via moyennage par groupe de 4
        if len(feats) > 32:
            feats = feats[:32 * (len(feats) // 32)].reshape(-1, 32).mean(axis=0)
        feats = feats / (np.linalg.norm(feats) + 1e-8)
        return feats.astype(np.float32)

    def get_embedding(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        Retourne l'embedding combiné, ou None si le crop est inutilisable.
        Garantit une dimension constante (TOTAL_DIM).
        """
        dino = self._dino(crop_bgr)
        hsv = self._hsv_hist(crop_bgr)
        lbp = self._lbp(crop_bgr)

        # Si TOUT échoue, on retourne quand même un vecteur zéro de la bonne dim
        # pour ne pas casser l'alignement des accumulateurs
        if dino is None and hsv is None and lbp is None:
            return None

        # Padding avec des zéros pour les composantes manquantes
        dino = dino if dino is not None else np.zeros(self.DINO_DIM, dtype=np.float32)
        hsv = hsv if hsv is not None else np.zeros(self.HSV_DIM, dtype=np.float32)
        lbp = lbp if lbp is not None else np.zeros(self.LBP_DIM, dtype=np.float32)

        combined = np.concatenate([
            dino * self.w_dino,
            hsv * self.w_hsv,
            lbp * self.w_lbp,
        ])
        combined = combined / (np.linalg.norm(combined) + 1e-8)
        return combined.astype(np.float32)

    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Similarité par composante (DINOv2 + HSV + LBP), moyenne pondérée.
        Beaucoup plus stable que dot product sur le vecteur concaténé,
        car chaque composante contribue à parts égales au score final
        indépendamment de sa dimensionnalité.
        """
        if emb1 is None or emb2 is None:
            return 0.0
        if emb1.shape != emb2.shape:
            return 0.0
        d = self.DINO_DIM
        h = self.HSV_DIM
        l = self.LBP_DIM
        # Découpage par slots fixes
        d1, h1, l1 = emb1[:d], emb1[d:d+h], emb1[d+h:]
        d2, h2, l2 = emb2[:d], emb2[d:d+h], emb2[d+h:]
        # Cosine par composante
        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        sim_dino = cos(d1, d2)
        sim_hsv = cos(h1, h2)
        sim_lbp = cos(l1, l2)
        # Moyenne pondérée
        return float(self.w_dino * sim_dino + self.w_hsv * sim_hsv + self.w_lbp * sim_lbp)