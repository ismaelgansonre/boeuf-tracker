"""
# Backward compatibility - imports from new structure
from models.analytics import Analytics, DetectionSample, get_analytics, init_analytics

__all__ = ["Analytics", "DetectionSample", "get_analytics", "init_analytics"]

Donnees collectees (toutes cross-session via web/data/history.json) :
1. **Historique FPS** : echantillonne toutes les 2s pour la courbe
2. **Repartition par robe-type** : compteurs par robe detectee
3. **Repartition par activite** : compteurs comportementaux
4. **Heatmap spatiale** : positions (x, y normalises 0-1) de chaque
   detection par animal, pour la carte de chaleur

Thread-safe via un Lock.
Thread leger qui echantillonne STATE periodiquement.
"""
import json
import os
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────
SAMPLE_INTERVAL = 2.0          # secondes entre deux echantillonnages
HISTORY_PATH = "web/data/history.json"
HEATMAP_GRID = 20              # grille NxN pour la heatmap (normalisee 0-1)
MAX_TIMELINE_EVENTS = 200      # limite pour eviter une croissance infinie


@dataclass
class DetectionSample:
    """Un echantillon de detection a un instant t."""
    frame: int
    proper_name: str                # "Marguerite"
    key: str                        # "Boeuf_001"
    breed: str                      # coat-type
    conf: float                     # confiance YOLO
    cx_norm: float                  # position X normalisee 0-1
    cy_norm: float                  # position Y normalisee 0-1
    behavior: str                   # 'grazing', 'walking', 'lying', ...
    source: str = ""                # nom de la video/source courante


@dataclass
class AnalyticsState:
    """Etat accumule pour le dashboard."""
    started_at: float = field(default_factory=time.time)
    fps_history: list = field(default_factory=list)        # [{t, fps}, ...]
    race_counts: dict = field(default_factory=dict)        # {robe-type: count}
    activity_counts: dict = field(default_factory=dict)    # {behavior: count}
    detection_samples: list = field(default_factory=list)   # pour heatmap
    timeline_events: list = field(default_factory=list)     # [{t, type, msg}, ...]

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "uptime": time.time() - self.started_at,
            "fps_history": self.fps_history[-300:],         # 10min @ 2s
            "race_counts": dict(self.race_counts),
            "activity_counts": dict(self.activity_counts),
            "detection_count": len(self.detection_samples),
            "timeline_events": self.timeline_events[-MAX_TIMELINE_EVENTS:],
        }

    def compute_profiles(self) -> dict:
        """Agrege les detection_samples en profils par bovin.

        Pour chaque bovin (cle proper_name), on calcule :
          - total_samples : nombre total de detections
          - videos : {nom_video: count} -> dans quelles videos il a ete vu
          - activities : {behavior: count} -> ce qu'il faisait
          - activities_pct : {behavior: pct} -> en pourcentage
          - first_seen / last_seen : fenetre temporelle
          - breed : derniere race connue
        """
        # {proper_name: {key, breed, samples, videos: {src: n}, acts: {beh: n}, frames: [min,max]}}
        raw: dict = defaultdict(lambda: {
            "key": "", "breed": "Indeterminee",
            "total": 0,
            "videos": defaultdict(int),
            "activities": defaultdict(int),
            "min_frame": None, "max_frame": None,
        })
        for s in self.detection_samples:
            name = s.get("proper_name") or s.get("key") or "?"
            r = raw[name]
            r["key"] = s.get("key", r["key"])
            breed = s.get("breed")
            if breed and breed != "Indeterminate":
                r["breed"] = breed
            r["total"] += 1
            src = s.get("source") or "inconnu"
            r["videos"][src] += 1
            act = s.get("behavior") or "active"
            r["activities"][act] += 1
            fr = s.get("frame")
            if fr is not None:
                r["min_frame"] = fr if r["min_frame"] is None else min(r["min_frame"], fr)
                r["max_frame"] = fr if r["max_frame"] is None else max(r["max_frame"], fr)

        # Conversion en format serialisable + pourcentages
        profiles = {}
        for name, r in raw.items():
            total = r["total"] or 1
            profiles[name] = {
                "key": r["key"],
                "breed": r["breed"],
                "total_samples": r["total"],
                "videos": dict(r["videos"]),
                "video_count": len(r["videos"]),
                "activities": dict(r["activities"]),
                "activities_pct": {k: round(v / total * 100, 1)
                                   for k, v in r["activities"].items()},
                "first_frame": r["min_frame"],
                "last_frame": r["max_frame"],
            }
        return profiles


class AnalyticsCollector:
    """Collecte periodique depuis STATE, persistee sur disque.

    Usage :
        collector = AnalyticsCollector(state_provider=lambda: STATE)
        collector.start()
        # ... plus tard ...
        data = collector.get_dashboard()
    """

    def __init__(self, state_provider, history_path: str = HISTORY_PATH):
        self._state_provider = state_provider
        self._path = history_path
        self.state = AnalyticsState()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Charge l'historique precedent pour avoir des donnees cross-session
        self._load_previous()

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="analytics")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._save()

    def _load_previous(self):
        """Charge history.json si present (cross-session)."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            self.state.fps_history = data.get("fps_history", [])
            self.state.race_counts = data.get("race_counts", {})
            self.state.activity_counts = data.get("activity_counts", {})
            # On NE charge PAS les detection_samples (trop lourd, que la session)
            # ni les timeline_events (cross-session OK)
            self.state.timeline_events = data.get("timeline_events", [])
            print(f"[analytics] historique charge: {len(self.state.fps_history)} fps samples, "
                  f"{len(self.state.race_counts)} races, "
                  f"{len(self.state.timeline_events)} timeline events")
        except Exception as e:
            print(f"[analytics] historique illisible, ignore: {e}")

    def _save(self):
        """Persiste l'etat sur disque."""
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        try:
            with open(self._path, "w") as f:
                json.dump(self.state.to_dict(), f)
        except Exception as e:
            print(f"[analytics] save failed: {e}")

    def _run(self):
        """Boucle d'echantillonnage."""
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception as e:
                print(f"[analytics] sample error: {e}")
            # Save toutes les ~10s pour eviter IO excessif
            if int(time.time()) % 10 < SAMPLE_INTERVAL:
                self._save()
            self._stop.wait(SAMPLE_INTERVAL)
        self._save()

    def _sample_once(self):
        """Echantillonne STATE une fois."""
        state = self._state_provider()
        now = time.time()

        with self._lock:
            # FPS
            fps = state.get("fps") or 0
            self.state.fps_history.append({"t": round(now - self.state.started_at, 1),
                                           "fps": round(fps, 2)})
            # Garde seulement les 300 derniers points (~10min a 2s)
            if len(self.state.fps_history) > 300:
                self.state.fps_history = self.state.fps_history[-300:]

            # Races (cohort actuelle)
            for animal in state.get("active_animals", []):
                breed = animal.get("breed", "Indeterminee")
                if breed and breed != "?":
                    self.state.race_counts[breed] = self.state.race_counts.get(breed, 0) + 1

            # Comportements
            for bh in state.get("behavior", []):
                action = bh.get("action", "?")
                self.state.activity_counts[action] = self.state.activity_counts.get(action, 0) + 1

            # Nouveaux events timeline (ceux pas encore vus)
            # On se base sur l'index dans la liste circulaire — on ajoute ce qui est nouveau
            current_events = state.get("events", [])
            for e in current_events:
                # Cle dedoublonnage simple
                if not any(t["msg"] == e for t in self.state.timeline_events[-50:]):
                    event_type = "NEW" if e.startswith("NEW") \
                                 else "MATCH" if e.startswith("MATCH") \
                                 else "CRASH" if e.startswith("CRASH") \
                                 else "INFO"
                    self.state.timeline_events.append({
                        "t": round(now - self.state.started_at, 1),
                        "type": event_type,
                        "msg": e,
                    })

    def push_detection(self, sample: DetectionSample):
        """Pousse un echantillon de detection (appele depuis la boucle video)."""
        with self._lock:
            self.state.detection_samples.append(asdict(sample))
            # Cap pour eviter explosion memoire
            if len(self.state.detection_samples) > 5000:
                # Decimation : on garde 1 sur 2
                self.state.detection_samples = self.state.detection_samples[::2]

    def get_dashboard(self) -> dict:
        """Retourne les donnees agregees pour le dashboard."""
        with self._lock:
            data = self.state.to_dict()
            # Distribution par race en pourcentages
            total_races = sum(data["race_counts"].values()) or 1
            data["race_pct"] = {
                k: round(v / total_races * 100, 1)
                for k, v in data["race_counts"].items()
            }
            # Total animaux uniques
            data["unique_animals"] = self._count_unique()
        # Map nom-de-race -> couleur hex (pour le chart donut et la legende).
        # Prend les swatches definis dans breed.BREEDS ; les robes-type HSV
        # de l'ancien format sont aussi couverts via breed.COAT_SWATCHES.
        try:
            from breed import BREEDS, COAT_SWATCHES
            data["breed_colors"] = {
                name: info.get("swatch", "#888888")
                for name, info in BREEDS.items()
            }
            data["breed_colors"].update(COAT_SWATCHES)
        except Exception:
            data["breed_colors"] = {}
        return data

    def get_profiles(self) -> dict:
        """Retourne les profils agregees par bovin pour l'onglet Statistiques.

        Format : {proper_name: {key, breed, total_samples, videos, video_count,
                  activities, activities_pct, first_frame, last_frame}}
        """
        with self._lock:
            return self.state.compute_profiles()

    def get_heatmap(self) -> dict:
        """Retourne les positions pour la heatmap.

        Format : grille HEATMAP_GRID x HEATMAP_GRID, valeurs = count de detections.
        On calcule aussi par robe-type pour superposer plusieurs couches.
        """
        with self._lock:
            samples = self.state.detection_samples[-2000:]  # recents
        grid = defaultdict(int)
        per_breed = defaultdict(lambda: defaultdict(int))
        max_dim = HEATMAP_GRID - 1
        for s in samples:
            cx = max(0, min(max_dim, int(s["cx_norm"] * max_dim)))
            cy = max(0, min(max_dim, int(s["cy_norm"] * max_dim)))
            grid[(cx, cy)] += 1
            b = s.get("breed", "?")
            per_breed[b][(cx, cy)] += 1
        # Conversion en liste pour JSON
        return {
            "grid_size": HEATMAP_GRID,
            "total_samples": len(samples),
            "global": [
                {"x": x, "y": y, "count": c}
                for (x, y), c in grid.items()
            ],
            "by_breed": {
                breed: [{"x": x, "y": y, "count": c} for (x, y), c in data.items()]
                for breed, data in per_breed.items()
            },
        }

    def _count_unique(self) -> int:
        """Compte les animaux uniques vus (via les noms propres dans la timeline)."""
        names = set()
        for e in self.state.timeline_events:
            msg = e["msg"]
            # Extrait le nom propre (premier mot si format "Nom (...)" ou "Nom  ...")
            # Format NEW: "NEW  Marguerite (Boeuf_001)  ..."
            # Format MATCH: "MATCH Colette (Boeuf_005)  ..."
            parts = msg.split()
            if len(parts) >= 2:
                # Le nom est toujours apres NEW/MATCH
                candidate = parts[1]
                # Retire les parentheses eventuelles
                candidate = candidate.split("(")[0]
                if candidate and candidate[0].isupper():
                    names.add(candidate)
        return len(names)


# Instance globale (singleton)
_instance: Optional[AnalyticsCollector] = None


def init(state_provider, path: str = HISTORY_PATH) -> AnalyticsCollector:
    """Initialise le collecteur global et le demarre."""
    global _instance
    _instance = AnalyticsCollector(state_provider, path)
    _instance.start()
    return _instance


def get() -> Optional[AnalyticsCollector]:
    return _instance
