"""
models/animal.py
----------------
Animal data models.
"""
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Animal:
    """Represents a identified cattle animal."""
    key: str                      # Stable ID (e.g., "Boeuf_001")
    name: str                     # Display name (e.g., "Marguerite")
    embedding: np.ndarray         # Re-ID embedding (464 dim)
    breed: str = "Indeterminee"   # Detected breed
    first_seen: str = ""          # ISO timestamp
    count: int = 0                # Number of detections
    last_seen: str = ""           # ISO timestamp
    confidence_sum: float = 0.0   # Sum of detection confidences

    @property
    def avg_confidence(self) -> float:
        return self.confidence_sum / self.count if self.count > 0 else 0.0

    def update(self, embedding: np.ndarray, confidence: float = 1.0, breed: str | None = None) -> None:
        """Update embedding with EMA and increment count."""
        from datetime import datetime
        self.count += 1
        self.last_seen = datetime.now().isoformat()
        self.confidence_sum += confidence
        if breed and breed != "Indeterminee":
            self.breed = breed
        # EMA update of embedding
        alpha = 0.5
        self.embedding = alpha * embedding + (1 - alpha) * self.embedding
        # Re-normalize
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = self.embedding / norm

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "breed": self.breed,
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "avg_confidence": self.avg_confidence,
        }


@dataclass
class AnimalTrack:
    """Track state for a detected animal in a video frame."""
    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: Optional[np.ndarray] = None  # Segmentation mask
    embedding: Optional[np.ndarray] = None
    breed: str = "Indeterminee"
    confidence: float = 0.0
    behavior_code: int = 3  # 0=couché, 1=pâture, 2=boit, 3=immobile, 4=marche, 5=court, 6=rué
    center_x: float = 0.0
    center_y: float = 0.0

    def __post_init__(self):
        """Calculate center from bbox."""
        x1, y1, x2, y2 = self.bbox
        self.center_x = (x1 + x2) / 2
        self.center_y = (y1 + y2) / 2

    @property
    def behavior_label(self) -> str:
        labels = ("couché", "pâture", "boit", "immobile", "marche", "court", "rué")
        return labels[self.behavior_code] if 0 <= self.behavior_code < len(labels) else "?"

    @property
    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Extract crop from frame."""
        x1, y1, x2, y2 = self.bbox
        return frame[y1:y2, x1:x2]

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "breed": self.breed,
            "confidence": self.confidence,
            "behavior": self.behavior_label,
            "cx": self.center_x,
            "cy": self.center_y,
        }
