"""
reid.py
-------
Extraction d'embeddings pour la ré-identification de bovins.

Combine TROIS signaux complémentaires :

1. **MegaDescriptor** (dim variable, typ. 768/1024/1536) : backbone
   spécialisé re-ID animale (WildlifeDatasets, entraîné sur ~30 datasets),
   bat DINOv2 et CLIP sur benchmarks re-ID (arXiv 2311.09118).
2. **HSV histogram** (48 dim) : couleur du pelage.
3. **LBP texture** (32 dim) : texture locale.

Le vecteur final = concaténation pondérée, L2-normalisé.

Design
------
- OOP découplé : trois FeatureExtractor autonomes + un CattleReID qui les compose.
- Testable : chaque extractor a une signature `(crop_bgr) -> np.ndarray | None`
  et une dimension connue. Facile à mocker.
- KISS : pas de fallback silencieux qui masque un bug ; on lève ou on retourne
  explicitement `None`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
import torch
from PIL import Image

from console import info, ok, warn


# ─────────────────────────────────────────────────────────────────
#  Interfaces
# ─────────────────────────────────────────────────────────────────
class FeatureExtractor(Protocol):
    """Contract minimal partagé par tous les extractors."""
    dim: int

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray | None: ...

    def extract_batch(self, crops_bgr: list[np.ndarray]) -> list[np.ndarray | None]: ...


# ─────────────────────────────────────────────────────────────────
#  Deep backbone : MegaDescriptor (timm)
# ─────────────────────────────────────────────────────────────────
class MegaDescriptorExtractor:
    """Backbone re-ID animale. Charge un modèle timm depuis HuggingFace Hub.

    Modèles disponibles :
        hf-hub:BVRA/MegaDescriptor-T-224   (~28M, dim 768, rapide)
        hf-hub:BVRA/MegaDescriptor-S-224   (~50M)
        hf-hub:BVRA/MegaDescriptor-B-224   (~87M)
        hf-hub:BVRA/MegaDescriptor-L-384   (~220M, dim 1536, précis)
    """

    def __init__(
        self,
        model_name: str = "hf-hub:BVRA/MegaDescriptor-T-224",
        device: str = "cpu",
        use_compile: bool = True,
    ):
        import timm  # lazy import — évite la dépendance dure au import du module

        self.device = device
        self.model_name = model_name

        info(f"[MegaDescriptor] Chargement de {model_name} sur {device}...")
        # num_classes=0 → renvoie l'embedding pooled, pas les logits
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval().to(device)

        # Preprocessing standard du modèle (résolution, mean/std)
        data_cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**data_cfg, is_training=False)
        self.input_size = data_cfg["input_size"][-1]

        # FP16 sur GPU (CUDA & MPS)
        self._half = False
        if device.startswith("cuda") or device == "mps":
            try:
                self.model = self.model.half()
                self._half = True
                info(f"[MegaDescriptor] FP16 actif sur {device}")
            except Exception as e:
                warn(f"[MegaDescriptor] FP16 impossible ({e}), fallback FP32")

        # torch.compile sur CUDA uniquement (MPS instable)
        if use_compile and device.startswith("cuda") and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                info("[MegaDescriptor] torch.compile actif")
            except Exception as e:
                warn(f"[MegaDescriptor] torch.compile échoue ({e})")

        # Warmup + probe de la dim de sortie
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.input_size, self.input_size, device=device)
            if self._half:
                dummy = dummy.half()
            out = self.model(dummy)
            self.dim = int(out.shape[-1])
        ok(f"[MegaDescriptor] Prêt (input={self.input_size}, dim={self.dim}, fp16={self._half})")

    @staticmethod
    def _crop_is_valid(crop_bgr: np.ndarray | None, min_side: int = 16) -> bool:
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        h, w = crop_bgr.shape[:2]
        return h >= min_side and w >= min_side

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return self.transform(pil)

    @torch.no_grad()
    def extract(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        if not self._crop_is_valid(crop_bgr):
            return None
        x = self._preprocess(crop_bgr).unsqueeze(0).to(self.device)
        if self._half:
            x = x.half()
        emb = self.model(x).flatten().float().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

    @torch.no_grad()
    def extract_batch(self, crops_bgr: list[np.ndarray]) -> list[np.ndarray | None]:
        if not crops_bgr:
            return []
        # Filtre les crops invalides tout en gardant l'index d'origine
        valid_idx: list[int] = []
        tensors: list[torch.Tensor] = []
        for i, c in enumerate(crops_bgr):
            if self._crop_is_valid(c):
                valid_idx.append(i)
                tensors.append(self._preprocess(c))

        out: list[np.ndarray | None] = [None] * len(crops_bgr)
        if not tensors:
            return out

        batch = torch.stack(tensors).to(self.device)
        if self._half:
            batch = batch.half()
        emb_batch = self.model(batch).float().cpu().numpy()
        emb_batch = emb_batch / (np.linalg.norm(emb_batch, axis=1, keepdims=True) + 1e-8)
        for k, idx in enumerate(valid_idx):
            out[idx] = emb_batch[k].astype(np.float32)
        return out


# ─────────────────────────────────────────────────────────────────
#  HSV histogram
# ─────────────────────────────────────────────────────────────────
class HSVHistExtractor:
    dim = 48  # 16 bins × 3 canaux

    def __init__(self, min_side: int = 32):
        self.min_side = min_side

    def extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < self.min_side or w < self.min_side:
            return None
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        return hist / (hist.sum() + 1e-8)

    def extract_batch(self, crops: list[np.ndarray]) -> list[np.ndarray | None]:
        return [self.extract(c) for c in crops]


# ─────────────────────────────────────────────────────────────────
#  LBP texture
# ─────────────────────────────────────────────────────────────────
class LBPExtractor:
    dim = 32
    _OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))

    def __init__(self, grid: int = 4, min_side: int = 32):
        self.grid = grid
        self.min_side = min_side

    @classmethod
    def _lbp_image(cls, gray: np.ndarray) -> np.ndarray:
        lbp = np.zeros((gray.shape[0] - 2, gray.shape[1] - 2), dtype=np.uint8)
        for i, (dy, dx) in enumerate(cls._OFFSETS):
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            lbp += ((gray[1:-1, 1:-1] > shifted[1:-1, 1:-1]) << i).astype(np.uint8)
        return lbp

    def _hist_grid(self, lbp: np.ndarray, bins: int = 8) -> np.ndarray:
        gh, gw = lbp.shape[0] // self.grid, lbp.shape[1] // self.grid
        feats = []
        for gy in range(self.grid):
            for gx in range(self.grid):
                cell = lbp[gy * gh:(gy + 1) * gh, gx * gw:(gx + 1) * gw]
                hist, _ = np.histogram(cell, bins=bins, range=(0, 256))
                hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
                feats.append(hist)
        return np.concatenate(feats)  # grid*grid*bins = 128

    def extract(self, crop_bgr: np.ndarray | None) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < self.min_side or w < self.min_side:
            return None
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        feats = self._hist_grid(gray)
        # Réduction 128 → 32 par moyennage de groupes de 4
        feats = feats.reshape(-1, self.dim).mean(axis=0)
        return (feats / (np.linalg.norm(feats) + 1e-8)).astype(np.float32)

    def extract_batch(self, crops: list[np.ndarray]) -> list[np.ndarray | None]:
        return [self.extract(c) for c in crops]


# ─────────────────────────────────────────────────────────────────
#  Composition : CattleReID
# ─────────────────────────────────────────────────────────────────
@dataclass
class ReIDWeights:
    """Poids relatifs des trois signaux (normalisés en interne)."""
    deep: float = 0.7
    hsv: float = 0.2
    lbp: float = 0.1

    def normalized(self) -> "ReIDWeights":
        s = self.deep + self.hsv + self.lbp
        return ReIDWeights(self.deep / s, self.hsv / s, self.lbp / s)


class CattleReID:
    """Assemble MegaDescriptor + HSV + LBP en un seul embedding L2-normalisé.

    L'ordre concatenation : [deep | hsv | lbp]. Les slots sont référencés par
    `slice_deep()`, `slice_hsv()`, `slice_lbp()` pour éviter les indices magiques.
    """

    HSV_DIM = HSVHistExtractor.dim
    LBP_DIM = LBPExtractor.dim

    def __init__(
        self,
        model_name: str = "hf-hub:BVRA/MegaDescriptor-T-224",
        device: str = "cpu",
        weights: ReIDWeights | None = None,
        deep_extractor: FeatureExtractor | None = None,
        use_compile: bool = True,
    ):
        """`deep_extractor` peut être injecté pour les tests (mock).
        Sinon on charge MegaDescriptor par défaut.
        """
        self.deep: FeatureExtractor = deep_extractor or MegaDescriptorExtractor(
            model_name=model_name, device=device, use_compile=use_compile,
        )
        self.hsv = HSVHistExtractor()
        self.lbp = LBPExtractor()

        w = (weights or ReIDWeights()).normalized()
        self.w_deep, self.w_hsv, self.w_lbp = w.deep, w.hsv, w.lbp

        self.DEEP_DIM = self.deep.dim
        self.TOTAL_DIM = self.DEEP_DIM + self.HSV_DIM + self.LBP_DIM

    # ─── Slots ─────────────────────────────────────────────────
    def slice_deep(self) -> slice: return slice(0, self.DEEP_DIM)
    def slice_hsv(self)  -> slice: return slice(self.DEEP_DIM, self.DEEP_DIM + self.HSV_DIM)
    def slice_lbp(self)  -> slice: return slice(self.DEEP_DIM + self.HSV_DIM, self.TOTAL_DIM)

    # ─── Composition d'un embedding ────────────────────────────
    def _compose(
        self, deep: np.ndarray | None, hsv: np.ndarray | None, lbp: np.ndarray | None,
    ) -> np.ndarray | None:
        if deep is None and hsv is None and lbp is None:
            return None
        deep = deep if deep is not None else np.zeros(self.DEEP_DIM, dtype=np.float32)
        hsv  = hsv  if hsv  is not None else np.zeros(self.HSV_DIM,  dtype=np.float32)
        lbp  = lbp  if lbp  is not None else np.zeros(self.LBP_DIM,  dtype=np.float32)
        combined = np.concatenate([
            deep * self.w_deep,
            hsv  * self.w_hsv,
            lbp  * self.w_lbp,
        ])
        return (combined / (np.linalg.norm(combined) + 1e-8)).astype(np.float32)

    def get_embedding(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        return self._compose(
            self.deep.extract(crop_bgr),
            self.hsv.extract(crop_bgr),
            self.lbp.extract(crop_bgr),
        )

    def get_embedding_batch(self, crops_bgr: list[np.ndarray]) -> list[np.ndarray | None]:
        if not crops_bgr:
            return []
        deeps = self.deep.extract_batch(crops_bgr)
        hsvs  = self.hsv.extract_batch(crops_bgr)
        lbps  = self.lbp.extract_batch(crops_bgr)
        return [self._compose(d, h, l) for d, h, l in zip(deeps, hsvs, lbps)]

    # ─── Similarité (moyenne pondérée des cosines par slot) ───
    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def compare(self, emb1: np.ndarray | None, emb2: np.ndarray | None) -> float:
        if emb1 is None or emb2 is None or emb1.shape != emb2.shape:
            return 0.0
        sd = self._cos(emb1[self.slice_deep()], emb2[self.slice_deep()])
        sh = self._cos(emb1[self.slice_hsv()],  emb2[self.slice_hsv()])
        sl = self._cos(emb1[self.slice_lbp()],  emb2[self.slice_lbp()])
        return float(self.w_deep * sd + self.w_hsv * sh + self.w_lbp * sl)


# ─── Backward-compat aliases (code appelant historique) ───────
CattleReID.DINO_DIM = property(lambda self: self.DEEP_DIM)  # type: ignore[attr-defined]
