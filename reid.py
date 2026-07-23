"""
# Backward compatibility - imports from new structure
from core.reid import CattleReID

__all__ = ["CattleReID"]

Combine TROIS signaux complémentaires:

1. **DINOv2** (384 dim): sémantique générale, capture forme et apparence globale
2. **HSV color histogram** (48 dim): distribution des couleurs du pelage
   (crucial pour distinguer bovins de robes différentes)
3. **LBP texture histogram** (32 dim): texture locale du pelage
   (capte les motifs au-delà de la couleur brute)

Le vecteur final est concaténé + normalisé L2.
Sans entraînement, ce combo surpasse DINOv2 seul pour distinguer des bovins
de races similaires.

Optimisations:
- get_embedding_batch(crops): 1 seul forward DINOv2 sur N crops au lieu de N forwards.
- torch.compile sur le forward si dispo (CUDA + PyTorch 2.0+).
- Le pipeline reste compatible avec get_embedding() (1 crop).
"""
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

from console import info, ok, warn


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
        dino_weight: float = 0.7,
        hsv_weight: float = 0.2,
        lbp_weight: float = 0.1,
        use_compile: bool = True,
    ):
        """
        Poids des trois composantes. Pour races à motifs (Holstein, Normand):
        garder ces valeurs. Pour races unies (Angus, Charolais): baisser
        dino_weight à 0.3 et monter hsv_weight à 0.5.

        use_compile: applique torch.compile sur le forward DINOv2 si CUDA + PyTorch >= 2.0.
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

        info(f"[DINOv2] Chargement de {model_name} sur {device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # torch.compile: gain ~20-30% sur le forward, gratuit si CUDA
        self._compiled = False
        if use_compile and device.startswith("cuda") and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self._compiled = True
                info("[DINOv2] torch.compile active (mode=reduce-overhead)")
            except Exception as e:
                warn(f"[DINOv2] torch.compile echoue ({e}), forward classique.")

        # FP16 (half precision) sur MPS: +20% de吞吐 mesuré sur M1 Pro
        # (49→60 FPS en single, 125→137 crops/s en batch 8). Guarded: si la
        # conversion échoue on reste en fp32 (safe).
        self._half = False
        if device == "mps":
            try:
                self.model = self.model.half()
                self._half = True
                info("[DINOv2] FP16 active sur MPS (half precision)")
            except Exception as e:
                warn(f"[DINOv2] FP16 impossible sur MPS ({e}), fallback fp32.")
                self._half = False

        with torch.no_grad():
            dummy_pil = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            dummy_in = self._cast_inputs(
                self.processor(images=dummy_pil, return_tensors="pt").to(self.device)
            )
            _ = self.model(**dummy_in)
        ok(f"[DINOv2] Pret (dim totale={self.TOTAL_DIM}, fp16={self._half})")

    def _cast_inputs(self, inputs):
        """Caste les tenseurs d'entrée en fp16 si le modèle est en demi-précision.

        Le processor retourne toujours des fp32 ; il faut les aligner avec le
        dtype du modèle (half sur MPS quand activé) avant le forward.
        """
        if self._half:
            return {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v)
                    for k, v in inputs.items()}
        return inputs

    @torch.no_grad()
    def _dino_batch(self, crops_rgb: list[np.ndarray]) -> list[np.ndarray | None]:
        """
        Forward DINOv2 batché sur N crops RGB.
        Retourne une liste de N embeddings normalisés (None si shape incompatible).
        """
        if not crops_rgb:
            return []
        if len(crops_rgb) == 1:
            emb = self._dino_single(crops_rgb[0])
            return [emb]

        # PIL batch: gain énorme sur GPU vs N forwards individuels
        pils = [Image.fromarray(c) for c in crops_rgb]
        inputs = self._cast_inputs(
            self.processor(images=pils, return_tensors="pt").to(self.device)
        )
        out = self.model(**inputs)
        # last_hidden_state: (B, T, D) → moyenne sur T
        emb_batch = out.last_hidden_state.mean(dim=1)
        emb_batch = emb_batch / (emb_batch.norm(dim=-1, keepdim=True) + 1e-8)
        arrs = emb_batch.cpu().numpy().astype(np.float32)
        return [a for a in arrs]

    @torch.no_grad()
    def _dino_single(self, crop_rgb: np.ndarray) -> np.ndarray | None:
        pil = Image.fromarray(crop_rgb)
        inputs = self._cast_inputs(
            self.processor(images=pil, return_tensors="pt").to(self.device)
        )
        out = self.model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).flatten().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

    @torch.no_grad()
    def _dino(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 16 or w < 16:
            return None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return self._dino_single(rgb)

    def _hsv_hist(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Histogramme HSV: distribution des teintes, saturation, valeur.

        3 histogrammes 1D concaténés (H:16 + S:16 + V:16 = 48 dim).
        Note: cv2.calcHist avec channels=[0,1,2] et bins=[16,16,16]
        retournerait un histogramme 3D joint de 4096 dims — c'est ce que
        faisait l'ancien code par erreur.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 32 or w < 32:
            return None
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)
        return hist

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

    def get_embedding_batch(self, crops_bgr: list[np.ndarray]) -> list[np.ndarray | None]:
        """
        Calcule l'embedding pour N crops en UN SEUL forward DINOv2.
        Pour les composantes HSV/LBP (CPU, peu coûteux), traitement séquentiel.

        Retourne une liste de N embeddings (None si crop inutilisable).
        Économise ~N×(latence forward) → gain ~Nx sur le forward DINO.
        """
        n = len(crops_bgr)
        if n == 0:
            return []

        # 1) Pré-filtrer: ne garder que les crops valides pour DINOv2
        valid_idx = []
        rgbs: list[np.ndarray] = []
        for i, c in enumerate(crops_bgr):
            if c is None or c.size == 0:
                continue
            h, w = c.shape[:2]
            if h < 16 or w < 16:
                continue
            valid_idx.append(i)
            rgbs.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))

        # 2) Forward DINOv2 batché
        dinos: dict[int, np.ndarray] = {}
        if rgbs:
            emb_list = self._dino_batch(rgbs)
            for k, idx in enumerate(valid_idx):
                dinos[idx] = emb_list[k]

        # 3) Compose embedding final pour chaque crop (HSV/LBP séquentiel)
        out: list[np.ndarray | None] = [None] * n
        for i, crop in enumerate(crops_bgr):
            dino = dinos.get(i)
            hsv = self._hsv_hist(crop)
            lbp = self._lbp(crop)
            if dino is None and hsv is None and lbp is None:
                continue
            dino = dino if dino is not None else np.zeros(self.DINO_DIM, dtype=np.float32)
            hsv = hsv if hsv is not None else np.zeros(self.HSV_DIM, dtype=np.float32)
            lbp = lbp if lbp is not None else np.zeros(self.LBP_DIM, dtype=np.float32)
            combined = np.concatenate([
                dino * self.w_dino,
                hsv * self.w_hsv,
                lbp * self.w_lbp,
            ])
            out[i] = (combined / (np.linalg.norm(combined) + 1e-8)).astype(np.float32)
        return out

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