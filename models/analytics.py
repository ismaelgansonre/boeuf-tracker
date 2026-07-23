"""
models/analytics.py
-------------------
Analytics accumulation for the dashboard.
"""
import json
import os
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─── Configuration ────────────────────────────────────────────────────────────
SAMPLE_INTERVAL = 2.0          # seconds between samplings
HISTORY_PATH = "web/data/history.json"
HEATMAP_GRID = 20              # NxN grid for heatmap (normalized 0-1)
MAX_TIMELINE_EVENTS = 200      # limit to prevent infinite growth


@dataclass
class DetectionSample:
    """A detection sample at time t."""
    frame: int
    proper_name: str                # "Marguerite"
    key: str                       # "Boeuf_001"
    breed: str                     # coat-type
    conf: float                    # YOLO confidence
    cx_norm: float                 # normalized X position 0-1
    cy_norm: float                 # normalized Y position 0-1
    behavior: str                  # 'grazing', 'walking', 'lying', ...
    source: str = ""                # current video/source name


@dataclass
class Analytics:
    """Accumulated analytics state for the dashboard."""
    sample_interval: float = SAMPLE_INTERVAL
    history_path: str = HISTORY_PATH
    heatmap_grid: int = HEATMAP_GRID
    max_timeline_events: int = MAX_TIMELINE_EVENTS
    
    _started_at: float = field(default_factory=time.time, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _fps_history: list = field(default_factory=list, init=False)
    _race_counts: dict = field(default_factory=dict, init=False)
    _activity_counts: dict = field(default_factory=dict, init=False)
    _detection_samples: list = field(default_factory=list, init=False)
    _timeline_events: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self._started_at = time.time()

    @property
    def uptime(self) -> float:
        return time.time() - self._started_at

    def sample_fps(self, fps: float) -> None:
        """Record FPS sample."""
        with self._lock:
            self._fps_history.append({"t": time.time(), "fps": fps})
            self._fps_history = self._fps_history[-300:]  # 10min @ 2s

    def sample_detection(self, sample: DetectionSample) -> None:
        """Record a detection sample."""
        with self._lock:
            self._detection_samples.append(asdict(sample))
            self._race_counts[sample.breed] = self._race_counts.get(sample.breed, 0) + 1
            self._activity_counts[sample.behavior] = self._activity_counts.get(sample.behavior, 0) + 1

    def add_event(self, event_type: str, message: str) -> None:
        """Add a timeline event."""
        with self._lock:
            self._timeline_events.append({
                "t": time.time(),
                "type": event_type,
                "msg": message,
            })
            self._timeline_events = self._timeline_events[-self.max_timeline_events:]

    def to_dict(self) -> dict:
        """Export analytics as dict."""
        with self._lock:
            return {
                "started_at": self._started_at,
                "uptime": self.uptime,
                "fps_history": self._fps_history[-300:],
                "race_counts": dict(self._race_counts),
                "activity_counts": dict(self._activity_counts),
                "detection_count": len(self._detection_samples),
                "timeline_events": self._timeline_events[-self.max_timeline_events:],
            }

    def compute_profiles(self) -> dict:
        """Aggregate detection samples into per-cattle profiles."""
        raw: dict = defaultdict(lambda: {
            "key": "", "breed": "Indeterminee",
            "total": 0,
            "videos": defaultdict(int),
            "activities": defaultdict(int),
            "min_frame": None, "max_frame": None,
        })
        for s in self._detection_samples:
            name = s.get("proper_name") or s.get("key") or "?"
            r = raw[name]
            r["key"] = s.get("key", r["key"])
            breed = s.get("breed")
            if breed and breed != "Indeterminee":
                r["breed"] = breed
            r["total"] += 1
            src = s.get("source") or "unknown"
            r["videos"][src] += 1
            act = s.get("behavior") or "active"
            r["activities"][act] += 1
            fr = s.get("frame")
            if fr is not None:
                if r["min_frame"] is None or fr < r["min_frame"]:
                    r["min_frame"] = fr
                if r["max_frame"] is None or fr > r["max_frame"]:
                    r["max_frame"] = fr
        profiles = {}
        for name, data in raw.items():
            acts = dict(data["activities"])
            total = data["total"]
            profiles[name] = {
                "key": data["key"],
                "breed": data["breed"],
                "total_samples": total,
                "videos": dict(data["videos"]),
                "activities": acts,
                "activities_pct": {k: round(v / total * 100, 1) for k, v in acts.items()},
                "min_frame": data["min_frame"],
                "max_frame": data["max_frame"],
            }
        return profiles

    def save_history(self, path: str | None = None) -> None:
        """Save history to JSON file."""
        path = path or self.history_path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump({
                    "analytics": self.to_dict(),
                    "profiles": self.compute_profiles(),
                }, f, indent=2)
        except Exception as e:
            pass  # Non-critical

    def load_history(self, path: str | None = None) -> bool:
        """Load history from JSON file. Returns True if successful."""
        path = path or self.history_path
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            # Could restore state here if needed
            return True
        except Exception:
            return False

    def reset(self) -> None:
        """Reset all analytics."""
        with self._lock:
            self._started_at = time.time()
            self._fps_history.clear()
            self._race_counts.clear()
            self._activity_counts.clear()
            self._detection_samples.clear()
            self._timeline_events.clear()

    def __repr__(self) -> str:
        return f"Analytics(uptime={self.uptime:.0f}s, detections={len(self._detection_samples)})"


# ─── Global singleton ─────────────────────────────────────────────────────────
_analytics_instance: Optional[Analytics] = None
_analytics_lock = threading.Lock()


def get_analytics() -> Analytics:
    """Get or create global Analytics instance."""
    global _analytics_instance
    with _analytics_lock:
        if _analytics_instance is None:
            _analytics_instance = Analytics()
        return _analytics_instance


def init_analytics() -> Analytics:
    """Initialize and return global Analytics instance."""
    global _analytics_instance
    with _analytics_lock:
        _analytics_instance = Analytics()
        return _analytics_instance
