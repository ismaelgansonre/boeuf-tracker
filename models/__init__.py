"""
models/__init__.py
------------------
Data models for Boeuf Tracker.
"""
from .animal import Animal, AnimalTrack
from .breed import BreedClassifier, BREEDS
from .analytics import Analytics, DetectionSample

__all__ = [
    "Animal",
    "AnimalTrack",
    "BreedClassifier", 
    "BREEDS",
    "Analytics",
    "DetectionSample",
]
