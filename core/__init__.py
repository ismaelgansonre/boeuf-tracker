"""
core/__init__.py
----------------
Core modules for Boeuf Tracker.
"""
from .detector import CattleDetector, CattleDetectorMLX
from .reid import CattleReID
from .database import EmbeddingDatabase
from .capture import CaptureManager

__all__ = [
    "CattleDetector",
    "CattleDetectorMLX", 
    "CattleReID",
    "EmbeddingDatabase",
    "CaptureManager",
]
