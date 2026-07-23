"""
config.py
---------
Configuration centralisée du projet Boeuf Tracker.
"""
import os
from dataclasses import dataclass, field


# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

DEFAULT_DB_PATH = "cattle_db.pkl"
DEFAULT_COUNTER_PATH = "names_counter.json"
DEFAULT_HISTORY_PATH = "web/data/history.json"


# ─── Model Defaults ──────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Configuration des modèles ML."""
    yolo_model: str = "yolo11s-seg.pt"
    dino_model: str = "facebook/dinov2-small"
    reid_engine: str | None = None  # MLX, PyTorch, etc.
    
    # Auto-detect MLX model if available
    @classmethod
    def auto_yolo(cls) -> str:
        mlx_path = os.path.join(PROJECT_ROOT, "yolo26s-seg.safetensors")
        return "yolo26s-seg.safetensors" if os.path.exists(mlx_path) else "yolo11s-seg.pt"


# ─── Detection Defaults ──────────────────────────────────────────────────────
@dataclass
class DetectionConfig:
    """Configuration de détection."""
    conf_threshold: float = 0.4
    reid_threshold: float = 0.70
    loop_threshold: float = 0.55
    loop_grace_frames: int = 60
    max_ema_updates: int = 30
    embed_every: int = 10
    imgsz: int = 640


# ─── Re-ID Weights ───────────────────────────────────────────────────────────
@dataclass
class ReIDConfig:
    """Poids des composantes Re-ID."""
    dino_weight: float = 0.7
    hsv_weight: float = 0.2
    lbp_weight: float = 0.1
    use_compile: bool = True


# ─── Analytics Defaults ──────────────────────────────────────────────────────
@dataclass
class AnalyticsConfig:
    """Configuration analytique."""
    sample_interval: float = 2.0
    history_path: str = DEFAULT_HISTORY_PATH
    heatmap_grid: int = 20
    max_timeline_events: int = 200


# ─── App Defaults ────────────────────────────────────────────────────────────
@dataclass
class AppConfig:
    """Configuration de l'application Flask."""
    host: str = "0.0.0.0"
    port: int = 8100
    ui_dir: str = "web/public"
    debug: bool = False


# ─── Device Config ───────────────────────────────────────────────────────────
import torch


@dataclass
class DeviceConfig:
    """Configuration du device (CPU/GPU)."""
    device: str = "auto"
    
    def resolve(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    @property
    def is_cuda(self) -> bool:
        return self.resolve().startswith("cuda")
    
    @property
    def is_mps(self) -> bool:
        return self.resolve() == "mps"


# ─── Palette de couleurs ─────────────────────────────────────────────────────
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


# ─── Global Config Singleton ─────────────────────────────────────────────────
@dataclass
class Config:
    """Configuration globale de l'application."""
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    app: AppConfig = field(default_factory=AppConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    
    @classmethod
    def from_args(cls, args) -> "Config":
        """Crée une config depuis les arguments CLI."""
        cfg = cls()
        if hasattr(args, 'yolo_model') and args.yolo_model:
            cfg.model.yolo_model = args.yolo_model
        if hasattr(args, 'dino_model') and args.dino_model:
            cfg.model.dino_model = args.dino_model
        if hasattr(args, 'threshold'):
            cfg.detection.reid_threshold = args.threshold
        if hasattr(args, 'conf'):
            cfg.detection.conf_threshold = args.conf
        if hasattr(args, 'device') and args.device != "auto":
            cfg.device.device = args.device
        if hasattr(args, 'db'):
            cfg.db_path = args.db
        return cfg


# Singleton global
config = Config()
