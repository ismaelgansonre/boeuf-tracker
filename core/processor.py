"""
core/processor.py
-----------------
Main detection loop: reads source, detects + tracks + identifies cattle,
annotates frame, updates STATE.

Auto-recovery on crash (loop restarts automatically).

Optimizations:
- Numba JIT on behavior computation (tight Python loop).
- Batch DINOv2 embeddings via reid.get_embedding_batch() for all new
  tracks in a frame in ONE forward pass.
"""
import os
import time
import threading
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

# Numba JIT
try:
    from numba import njit as _njit
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False

    def _njit(*args, **kwargs):
        def deco(fn): return fn
        return deco
    try:
        from utils.console import warn as _warn_fallback
        _warn_fallback(
            "[Numba] numba not installed — _total_displacement and "
            "_classify_behavior run in pure Python (×5-10 slower). "
            "Install with: pip install numba"
        )
    except Exception:
        pass


@_njit(cache=True, fastmath=True)
def _total_displacement(pts_x, pts_y):
    """Sum of Euclidean distances between consecutive points. Numba JIT."""
    s = 0.0
    for i in range(1, pts_x.shape[0]):
        dx = pts_x[i] - pts_x[i - 1]
        dy = pts_y[i] - pts_y[i - 1]
        s += (dx * dx + dy * dy) ** 0.5
    return s


@_njit(cache=True, fastmath=True)
def _classify_behavior(speed: float, aspect: float, rel_y: float,
                       immobile_dur: float) -> int:
    """
    Returns action code (0..6). Numba JIT.
    0=lying, 1=grazing, 2=drinking, 3=immobile, 4=walking, 5=running, 6=charging.
    """
    if aspect > 1.7 and speed < 5.0 and immobile_dur > 3.0:
        return 0
    if speed < 6.0 and aspect > 1.4:
        return 1
    if speed < 4.0 and aspect > 1.3 and rel_y > 0.6:
        return 2
    if speed < 5.0:
        return 3
    if speed < 25.0:
        return 4
    if speed < 80.0:
        return 5
    return 6


_BEHAVIOR_LABELS = ("couché", "pâture", "boit", "immobile", "marche", "court", "rué")

# MLX availability check
_mlx_available = False
try:
    import os as _os
    _mlx_path = _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), "..", "yolo26s-seg.safetensors"
    )
    _mlx_available = _os.path.exists(_mlx_path)
except Exception:
    pass


# ─── Imports requiring path setup ──────────────────────────────────────────────
import sys
sys.path.insert(0, '..')
from utils.console import info, ok, warn, err, dbg, evt, loop as log_loop
from utils.state import STATE
from utils.names import get_name_generator, next_bovin_key
from models.analytics import get_analytics, DetectionSample
from models.breed import classify_breed
from core.detector import CattleDetector, CattleDetectorMLX
from core.reid import CattleReID
from core.database import EmbeddingDatabase
from core.capture import CaptureManager, source_label


def resolve_device(device: str) -> str:
    """Resolve device string to actual device."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def annotate_frame(annotated, masks_data, det_idx, x1, y1, x2, y2, color):
    """Draw segmentation mask if available, else rectangle."""
    if masks_data is not None and det_idx < len(masks_data):
        mask = masks_data[det_idx]
        if mask is not None:
            color_array = np.array(color, dtype=np.uint8)
            mask_bool = mask.astype(bool)
            annotated[mask_bool] = annotated[mask_bool] * 0.6 + color_array * 0.4
            return
    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)


class VideoProcessor:
    """Main video processing pipeline."""

    def __init__(
        self,
        source: str,
        yolo_model: str = "yolo11s-seg.pt",
        dino_model: str = "facebook/dinov2-small",
        threshold: float = 0.70,
        db_path: str = "cattle_db.pkl",
        conf: float = 0.4,
        device: str = "auto",
        mlx: bool = False,
        loop_threshold: float = 0.55,
        loop_grace_frames: int = 60,
        max_updates: int = 30,
    ):
        self.source = source
        self.yolo_model = yolo_model
        self.dino_model = dino_model
        self.threshold = threshold
        self.db_path = db_path
        self.conf = conf
        self.device = resolve_device(device)
        self.mlx = mlx and _mlx_available
        self.loop_threshold = loop_threshold
        self.loop_grace_frames = loop_grace_frames
        self.max_updates = max_updates

        # State
        self.detector: Optional[CattleDetector] = None
        self.reid: Optional[CattleReID] = None
        self.db: Optional[EmbeddingDatabase] = None
        self.name_gen = get_name_generator()
        self.analytics = get_analytics()

        # Track state
        self.track_id_to_name: dict[int, str] = {}
        self.track_emb_accum: dict[int, list] = {}
        self.track_update_count: dict[int, int] = {}
        self.track_history: dict[int, list] = {}
        self._loop_grace_counter = 0
        self._is_video = isinstance(source, str) and not source.isdigit()

        # Performance
        self.frame_count = 0
        self.fps = 0.0
        self.last_time = time.time()

    def setup(self) -> None:
        """Initialize models and database."""
        info(f"[Processor] Setting up on {self.device}...")
        self.detector = CattleDetector(
            model_name=self.yolo_model,
            device=self.device,
        ) if not self.mlx else CattleDetectorMLX(
            model_name=self.yolo_model,
        )
        self.reid = CattleReID(
            model_name=self.dino_model,
            device=self.device,
        )
        self.db = EmbeddingDatabase(path=self.db_path, reid_engine=self.reid)
        self.db.validate_dim(self.reid.TOTAL_DIM)
        self.name_gen.build_from_db(self.db.animals)
        STATE["yolo_model_current"] = self.yolo_model
        STATE["device"] = self.device
        ok(f"[Processor] Ready (device={self.device}, mlx={self.mlx})")

    def process_frame(self, frame) -> None:
        """Process a single frame."""
        t0 = time.time()
        results = self.detector.detect(frame, conf=self.conf)
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return

        # Get masks if available
        masks = results.masks

        h, w = frame.shape[:2]

        for i, box in enumerate(boxes):
            track_id = int(box.id) if box.id is not None else i
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])

            # Get crop for Re-ID
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue

            # Update track history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            history = self.track_history[track_id]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            history.append((cx, cy, time.time()))
            if len(history) > 30:
                history.pop(0)

            # Behavior classification
            behavior_code = 3  # default immobile
            if len(history) >= 2:
                pts = np.array(history)
                speed = _total_displacement(pts[:, 0], pts[:, 1])
                aspect = (y2 - y1) / max(x2 - x1, 1)
                rel_y = cy / h
                immobile_dur = time.time() - history[0][2] if history else 0
                behavior_code = _classify_behavior(speed, aspect, rel_y, immobile_dur)

            # Get or create name
            if track_id not in self.track_id_to_name:
                # Try to match existing
                emb = self.reid.get_embedding(crop)
                if emb is not None:
                    effective_threshold = self.loop_threshold if self._loop_grace_counter > 0 else self.threshold
                    match_name, sim = self.db.match(emb, threshold=effective_threshold)
                    if match_name:
                        self.track_id_to_name[track_id] = match_name
                        evt(f"MATCH  {match_name}  (sim={sim:.2f})")
                    else:
                        key = next_bovin_key()
                        name = self.name_gen.get_name(key)
                        self.track_id_to_name[track_id] = name
                        self.db.add(name, emb)
                        ok(f"New: {name} ({key})")
                else:
                    name = f"Track_{track_id}"
                    self.track_id_to_name[track_id] = name
                self.track_emb_accum[track_id] = []
                self.track_update_count[track_id] = 0

            name = self.track_id_to_name.get(track_id, f"Track_{track_id}")

            # Embedding update every ~10 frames
            if self.frame_count % 10 == 0 and track_id in self.track_emb_accum:
                emb = self.reid.get_embedding(crop)
                if emb is not None:
                    self.track_emb_accum[track_id].append(emb)
                    if len(self.track_emb_accum[track_id]) >= 3:
                        avg_emb = np.mean(self.track_emb_accum[track_id], axis=0)
                        self.track_emb_accum[track_id] = []
                        if self.track_update_count.get(track_id, 0) < self.max_updates:
                            self.db.add(name, avg_emb)
                            self.track_update_count[track_id] = self.track_update_count.get(track_id, 0) + 1

            # Breed classification
            breed, _ = classify_breed(crop)

            # Annotate
            color = self._color_for_name(name)
            annotate_frame(frame, masks.data if masks is not None else None, i, x1, y1, x2, y2, color)
            cv2.putText(frame, name, (int(x1), int(y1) - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Update STATE
            self._update_active_animals(track_id, name, breed, conf, cx, cy, behavior_code)

            # Analytics
            self.analytics.sample_detection(DetectionSample(
                frame=self.frame_count,
                proper_name=name,
                key=name,
                breed=breed,
                conf=conf,
                cx_norm=cx / w,
                cy_norm=cy / h,
                behavior=_BEHAVIOR_LABELS[behavior_code],
                source=STATE.get("source_label", ""),
            ))

        # FPS calculation
        self.frame_count += 1
        elapsed = time.time() - t0
        self.fps = 1.0 / elapsed if elapsed > 0 else 0
        STATE["fps"] = self.fps
        STATE["frame_count"] = self.frame_count

    def _update_active_animals(self, track_id, name, breed, conf, cx, cy, behavior_code):
        with STATE["frame_lock"]:
            active = STATE["active_animals"]
            # Update or add
            found = False
            for a in active:
                if a["track_id"] == track_id:
                    a.update({
                        "name": name, "breed": breed, "conf": conf,
                        "cx": cx, "cy": cy, "behavior": _BEHAVIOR_LABELS[behavior_code],
                    })
                    found = True
                    break
            if not found:
                active.append({
                    "track_id": track_id, "name": name, "breed": breed,
                    "conf": conf, "cx": cx, "cy": cy,
                    "behavior": _BEHAVIOR_LABELS[behavior_code],
                })
            STATE["active_animals"] = active

    def _color_for_name(self, name: str):
        from utils.state import color_for_name
        return color_for_name(name)

    def run(self) -> None:
        """Main processing loop."""
        self.setup()
        STATE["started_at"] = datetime.now().isoformat()

        with CaptureManager(self.source) as cap:
            if not cap.is_opened:
                err(f"[Processor] Cannot open source: {self.source}")
                return

            STATE["source"] = str(self.source)
            STATE["source_label"] = source_label(self.source)

            while True:
                ret, frame = cap.read()
                if not ret:
                    if self._is_video:
                        break  # End of video
                    continue

                try:
                    self.process_frame(frame)
                except Exception as e:
                    err(f"[Processor] Frame error: {e}")
                    continue

                # Encode for streaming
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMPORTANCE, 85])
                with STATE["frame_lock"]:
                    STATE["frame_jpg"] = jpg.tobytes()

                # Check for source change
                desired = STATE.get("desired_source")
                if desired and desired != self.source:
                    info(f"[Processor] Source change: {desired}")
                    self.source = desired
                    cap.release()
                    self._reset_state()
                    cap = CaptureManager(self.source)
                    if not cap.is_opened:
                        err(f"[Processor] Cannot open new source: {desired}")
                        break
                    STATE["source"] = str(desired)
                    STATE["source_label"] = source_label(desired)

        ok("[Processor] Processing complete")

    def _reset_state(self) -> None:
        """Reset state for new source."""
        self.track_id_to_name.clear()
        self.track_emb_accum.clear()
        self.track_update_count.clear()
        self.track_history.clear()
        self.frame_count = 0
        STATE.reset_for_new_source()


# ─── Thread management ────────────────────────────────────────────────────────
_processor_thread: Optional[threading.Thread] = None
_processor_instance: Optional[VideoProcessor] = None


def start_detection_thread(
    source: str,
    yolo_model: str = "yolo11s-seg.pt",
    dino_model: str = "facebook/dinov2-small",
    threshold: float = 0.70,
    db_path: str = "cattle_db.pkl",
    conf: float = 0.4,
    device: str = "auto",
    mlx: bool = False,
    loop_threshold: float = 0.55,
    loop_grace_frames: int = 60,
    max_updates: int = 30,
) -> None:
    """Start detection in background thread."""
    global _processor_thread, _processor_instance

    if _processor_thread is not None and _processor_thread.is_alive():
        warn("[Processor] Thread already running")
        return

    _processor_instance = VideoProcessor(
        source=source,
        yolo_model=yolo_model,
        dino_model=dino_model,
        threshold=threshold,
        db_path=db_path,
        conf=conf,
        device=device,
        mlx=mlx,
        loop_threshold=loop_threshold,
        loop_grace_frames=loop_grace_frames,
        max_updates=max_updates,
    )
    _processor_thread = threading.Thread(
        target=_processor_instance.run,
        daemon=True,
        name="video-processor",
    )
    _processor_thread.start()
    info("[Processor] Detection thread started")
