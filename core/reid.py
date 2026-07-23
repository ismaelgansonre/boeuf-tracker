"""
core/reid.py
------------
Embedding extraction for cattle re-identification.

Combines THREE complementary signals:
1. DINOv2 (384 dim): general semantics, captures form and global appearance
2. HSV color histogram (48 dim): coat color distribution
3. LBP texture histogram (32 dim): local coat texture

The final vector is concatenated + L2 normalized.
"""
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

import sys
sys.path.insert(0, '..')
from utils.console import info, ok, warn


class CattleReID:
    """Cattle re-identification engine."""
    
    # Embedding dimensions
    DINO_DIM = 384   # dinov2-small
    HSV_DIM = 48     # 16 per channel (H, S, V)
    LBP_DIM = 32     # local texture
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
        self.device = device
        self.dino_weight = dino_weight
        self.hsv_weight = hsv_weight
        self.lbp_weight = lbp_weight

        # Normalize weights to sum to 1
        s = dino_weight + hsv_weight + lbp_weight
        self.w_dino = dino_weight / s
        self.w_hsv = hsv_weight / s
        self.w_lbp = lbp_weight / s

        info(f"[DINOv2] Loading {model_name} on {device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # torch.compile for ~20-30% speedup on CUDA
        self._compiled = False
        if use_compile and device.startswith("cuda") and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                self._compiled = True
                info("[DINOv2] torch.compile active (mode=reduce-overhead)")
            except Exception as e:
                warn(f"[DINOv2] torch.compile failed ({e})")

        # FP16 on MPS: +20% throughput
        self._half = False
        if device == "mps":
            try:
                self.model = self.model.half()
                self._half = True
                info("[DINOv2] FP16 active on MPS")
            except Exception as e:
                warn(f"[DINOv2] FP16 failed on MPS ({e}), fallback fp32")
                self._half = False

        # Warmup
        with torch.no_grad():
            dummy_pil = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            dummy_in = self._cast_inputs(
                self.processor(images=dummy_pil, return_tensors="pt").to(self.device)
            )
            _ = self.model(**dummy_in)
        ok(f"[DINOv2] Ready (dim={self.TOTAL_DIM}, fp16={self._half})")

    def _cast_inputs(self, inputs):
        """Cast to half if needed."""
        if self._half:
            return inputs
        return inputs

    def get_embedding(self, crop: np.ndarray) -> np.ndarray | None:
        """Get embedding for a single crop."""
        if crop is None or crop.size == 0:
            return None
        try:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                inputs = self._cast_inputs(
                    self.processor(images=pil, return_tensors="pt").to(self.device)
                )
                outputs = self.model(**inputs)
                dino_emb = outputs.last_hidden_state[:, 0].squeeze().cpu().numpy()
            hsv_hist = self._compute_hsv(crop)
            lbp_hist = self._compute_lbp(crop)
            combined = np.concatenate([
                self.w_dino * dino_emb,
                self.w_hsv * hsv_hist,
                self.w_lbp * lbp_hist,
            ])
            norm = np.linalg.norm(combined)
            return combined / norm if norm > 0 else None
        except Exception as e:
            warn(f"[DINOv2] Embedding failed: {e}")
            return None

    def get_embedding_batch(self, crops: list) -> list:
        """Get embeddings for multiple crops in one forward pass."""
        if not crops:
            return []
        valid = [c for c in crops if c is not None and c.size > 0]
        if not valid:
            return [None] * len(crops)
        try:
            pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in valid]
            with torch.no_grad():
                inputs = self._cast_inputs(
                    self.processor(images=pil_images, return_tensors="pt").to(self.device)
                )
                outputs = self.model(**inputs)
                dino_embs = outputs.last_hidden_state[:, 0].squeeze().cpu().numpy()
                if len(dino_embs.shape) == 1:
                    dino_embs = dino_embs.reshape(1, -1)
            hsv_batch = np.stack([self._compute_hsv(c) for c in valid])
            lbp_batch = np.stack([self._compute_lbp(c) for c in valid])
            combined = np.concatenate([
                self.w_dino * dino_embs,
                self.w_hsv * hsv_batch,
                self.w_lbp * lbp_batch,
            ], axis=1)
            norms = np.linalg.norm(combined, axis=1, keepdims=True)
            combined = combined / (norms + 1e-8)
            result = []
            idx = 0
            for c in crops:
                if c is None or c.size == 0:
                    result.append(None)
                else:
                    result.append(combined[idx])
                    idx += 1
            return result
        except Exception as e:
            warn(f"[DINOv2] Batch embedding failed: {e}")
            return [None] * len(crops)

    @staticmethod
    def _compute_hsv(crop: np.ndarray) -> np.ndarray:
        """Compute HSV histogram (48 dim)."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = []
        for i, (channel, bins) in enumerate([(hsv[:,:,0], 16), (hsv[:,:,1], 16), (hsv[:,:,2], 16)]):
            h = cv2.calcHist([channel], [0], None, [bins], [0, 256])
            h = h.flatten() / (h.sum() + 1e-8)
            hist.extend(h)
        return np.array(hist, dtype=np.float32)

    @staticmethod
    def _compute_lbp(crop: np.ndarray) -> np.ndarray:
        """Compute LBP histogram (32 dim)."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64))
        lbp = np.zeros_like(small, dtype=np.uint8)
        for i in range(1, small.shape[0]-1):
            for j in range(1, small.shape[1]-1):
                center = small[i, j]
                code = 0
                code |= (small[i-1, j-1] >= center) << 7
                code |= (small[i-1, j] >= center) << 6
                code |= (small[i-1, j+1] >= center) << 5
                code |= (small[i, j+1] >= center) << 4
                code |= (small[i+1, j+1] >= center) << 3
                code |= (small[i+1, j] >= center) << 2
                code |= (small[i+1, j-1] >= center) << 1
                code |= (small[i, j-1] >= center) << 0
                lbp[i, j] = code
        hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        return hist

    def __repr__(self) -> str:
        return f"CattleReID(device={self.device}, dim={self.TOTAL_DIM})"
