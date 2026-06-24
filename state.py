"""
state.py
--------
État global partagé entre le thread de détection et les routes Flask.
Inclut un JSON encoder numpy-safe.
"""
import threading
import numpy as np
from flask.json.provider import DefaultJSONProvider


class NumpyJSONProvider(DefaultJSONProvider):
    """Sérialise les types numpy vers JSON sans planter."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


# Palette de 10 couleurs (BGR pour OpenCV)
PALETTE = [
    (16, 185, 129),    # vert
    (59, 130, 246),    # bleu
    (245, 158, 11),    # orange
    (236, 72, 153),    # rose
    (139, 92, 246),    # violet
    (34, 211, 238),    # cyan
    (250, 204, 21),    # jaune
    (248, 113, 113),   # rouge clair
    (52, 211, 153),    # vert clair
    (96, 165, 250),    # bleu clair
]


def color_for_name(name: str):
    """Couleur stable par nom (hash polynomial -> index palette)."""
    h = 0
    for c in name:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return PALETTE[h % len(PALETTE)]


STATE = {
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
    "source": "",
    "source_label": "",
    "current_source_path": None,
    "desired_source": None,
    "desired_device": None,
}


def reset_for_new_source():
    """À appeler quand la source change (webcam ↔ vidéo)."""
    with STATE["frame_lock"]:
        STATE["active_animals"] = []
        STATE["behavior"] = []
    STATE["track_history"].clear()
    STATE["events"].insert(0, "TRACKER reset")
    STATE["events"] = STATE["events"][:30]