"""
core/detector.py
----------------
Wrapper around YOLOv11 (Ultralytics) with instance segmentation.
task=segment produces per-pixel masks following the exact cow silhouette.

FP16 (half precision) enabled by default on CUDA.
"""
import numpy as np
import torch
from ultralytics import YOLO

import sys
sys.path.insert(0, '..')
from utils.console import info, ok, warn


class CattleDetector:
    """YOLO-based cattle detector with segmentation."""
    
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
        # FP16 only on CUDA (CPU doesn't support half efficiently)
        if half is None:
            half = device.startswith("cuda") or device == "mps"
        self.half = half

        info(f"[YOLO] Loading {model_name} on {device} (FP16={half})...")
        self.model = YOLO(model_name)

        # FP16 conversion once at startup
        if self.half and self.device.startswith("cuda"):
            try:
                self.model.model.half()
            except Exception as e:
                warn(f"[YOLO] FP16 failed ({e}), falling back to FP32")
                self.half = False

        # Warmup
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
        ok(f"[YOLO] Model {model_name} ready ({device}, FP16={self.half})")

    def detect(self, frame, persist: bool = True, conf: float = 0.4, imgsz: int = 640):
        """Detect + track + segment cattle. Returns results[0]."""
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

    def __repr__(self) -> str:
        return f"CattleDetector(model={self.model}, device={self.device})"


class CattleDetectorMLX:
    """
    YOLO26 on MLX (Apple Metal GPU) with native ByteTrack.
    ~2.6× faster than YOLO11s on PyTorch MPS for Apple Silicon.
    """

    def __init__(
        self,
        model_name: str = "yolo26s-seg.safetensors",
        device: str = "mlx",
    ):
        self.device = device
        self.model_name = model_name
        info(f"[YOLO-MLX] Loading {model_name} on {device}...")
        self.model = YOLO(model_name)
        ok(f"[YOLO-MLX] Model {model_name} ready (MLX)")

    def detect(self, frame, persist: bool = True, conf: float = 0.4, imgsz: int = 640):
        """Detect + track with MLX."""
        return self.model.track(
            frame,
            classes=[self.COW_CLASS_ID],
            persist=persist,
            device=self.device,
            verbose=False,
            conf=conf,
            imgsz=imgsz,
        )[0]

    def __repr__(self) -> str:
        return f"CattleDetectorMLX(model={self.model_name})"
