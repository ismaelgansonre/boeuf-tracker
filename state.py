"""
# Backward compatibility - imports from new structure
from utils.state import AppState, STATE, color_for_name, reset_for_new_source

__all__ = ["AppState", "STATE", "color_for_name", "reset_for_new_source", "NumpyJSONProvider"]
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
    "_track_behavior_hist": {},  # {track_id: [action_codes]} — lissage temporel
    "source": "",
    "source_label": "",
    "current_source_path": None,
    "desired_source": None,
    "desired_device": None,
    "desired_imgsz": None,        # nouvelle demande de resolution YOLO
    "desired_embed_every": None,  # nouvelle frequence d'embedding
    "desired_threshold": None,    # nouveau seuil cosine
    "desired_conf": None,         # nouvelle confiance YOLO
    "yolo_model_current": "",     # modele YOLO actif
    "imgsz_current": 640,
    "embed_every_current": 10,
    "threshold_current": 0.70,
    "conf_current": 0.4,
    "ui_dir": "web/public",       # repertoire de l'UI statique (sert sans Bun)
    "models_available": [
        # YOLO26 MLX (Metal GPU Apple Silicon - recommandee)
        "yolo26s-seg.safetensors",
        # YOLO11 standard
        "yolo11n.pt", "yolo11n-seg.pt",
        "yolo11s.pt", "yolo11s-seg.pt",
        "yolo11m.pt", "yolo11m-seg.pt",
        "yolo11l.pt", "yolo11l-seg.pt",
        "yolo11x.pt", "yolo11x-seg.pt",
    ],
}


def reset_for_new_source():
    """À appeler quand la source change (webcam ↔ vidéo)."""
    with STATE["frame_lock"]:
        STATE["active_animals"] = []
        STATE["behavior"] = []
    STATE["track_history"].clear()
    STATE["events"].insert(0, "TRACKER reset")
    STATE["events"] = STATE["events"][:30]