"""
utils/state.py
--------------
Global shared state between detection thread and Flask routes.
"""
import threading
from typing import Any

from config import PALETTE


class AppState:
    """Thread-safe application state."""

    def __init__(self):
        self._state: dict = {
            "frame_jpg": None,
            "frame_lock": threading.Lock(),
            "fps": 0.0,
            "device": "cpu",
            "started_at": None,
            "frame_count": 0,
            "active_animals": [],
            "events": [],
            "behavior": [],
            "track_history": {},
            "_track_behavior_hist": {},
            "source": "",
            "source_label": "",
            "current_source_path": None,
            "desired_source": None,
            "desired_device": None,
            "desired_imgsz": None,
            "desired_embed_every": None,
            "desired_threshold": None,
            "desired_conf": None,
            "yolo_model_current": "",
            "imgsz_current": 640,
            "embed_every_current": 10,
            "threshold_current": 0.70,
            "conf_current": 0.4,
            "ui_dir": "web/public",
            "models_available": [
                "yolo26s-seg.safetensors",
                "yolo11n.pt", "yolo11n-seg.pt",
                "yolo11s.pt", "yolo11s-seg.pt",
                "yolo11m.pt", "yolo11m-seg.pt",
                "yolo11l.pt", "yolo11l-seg.pt",
                "yolo11x.pt", "yolo11x-seg.pt",
            ],
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def update(self, **kwargs) -> None:
        self._state.update(kwargs)

    def __getitem__(self, key: str) -> Any:
        return self._state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._state[key] = value

    def acquire_lock(self, key: str) -> threading.Lock:
        """Get or create a lock for a key."""
        if key not in self._state:
            self._state[key] = threading.Lock()
        return self._state[key]

    def reset_for_new_source(self) -> None:
        """Call when source changes (webcam ↔ video)."""
        with self._state["frame_lock"]:
            self._state["active_animals"] = []
            self._state["behavior"] = []
        self._state["track_history"].clear()
        self._state["events"].insert(0, "TRACKER reset")
        self._state["events"] = self._state["events"][:30]


def color_for_name(name: str):
    """Stable color per name (polynomial hash -> palette index)."""
    h = 0
    for c in name:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return PALETTE[h % len(PALETTE)]


def reset_for_new_source() -> None:
    """Reset state for new source."""
    global STATE
    STATE.reset_for_new_source()


# ─── Global singleton ─────────────────────────────────────────────────────────
STATE = AppState()
