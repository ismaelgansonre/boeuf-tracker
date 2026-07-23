"""
api/__init__.py
---------------
Flask API for Boeuf Tracker.
"""
from .app import create_app

__all__ = ["create_app"]
