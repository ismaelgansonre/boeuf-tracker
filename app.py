"""
app.py
------
Serveur Flask minimal. Démarre le thread de détection et expose:
- /                          : UI
- /video_feed                : MJPEG stream
- /api/stats                 : JSON stats
- /api/upload-video          : upload vidéo
- /api/source/webcam         : bascule webcam
- /api/device                : change device
- /api/devices               : liste devices
- /api/diag                  : diagnostic complet
- /api/db/reset              : reset base de données
"""
import argparse
import os

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from state import STATE, NumpyJSONProvider
from processor import start_detection_thread, resolve_device, _mlx_available
from console import banner as log_banner, info, ok, warn, err

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _pick_default_source() -> str:
    """
    Choisit la source par défaut:
    1. Première vidéo .mp4/.mov/.avi à la racine du projet
    2. Sinon '0' (webcam)
    Permet de lancer `python app.py` sans paramètre quand des vidéos
    de test sont présentes à la racine.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
    try:
        for entry in sorted(os.listdir(project_root)):
            full = os.path.join(project_root, entry)
            if os.path.isfile(full) and entry.lower().endswith(video_exts):
                return os.path.abspath(full)
    except OSError:
        pass
    return "0"


def parse_args():
    """
    Défauts optimisés pour GTX 1660 Ti (6 GB) — équilibre perf / détection.
    Lance simplement:    python app.py
    Override possible via CLI (voir PERF_GUIDE_1660TI.md).

    Source par défaut: si une vidéo est présente à la racine du projet,
    elle est utilisée automatiquement. Sinon, webcam ('0').
    """
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default=_pick_default_source(),
                   help="'0' = webcam | chemin vidéo | URL RTSP. "
                        "Défaut: 1ère vidéo trouvée à la racine du projet, "
                        "sinon '0'.")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8100,
                   help="Port du worker Python (Bun proxy dessus sur :8000).")
    # --- Modèles ---
    # Auto-sélection : YOLO26 MLX sur Apple Silicon (Metal GPU natif, ~26 FPS),
    # sinon YOLO11s-seg PyTorch (MPS/CUDA/CPU).
    import os as _os
    _has_mlx_model = _os.path.exists(
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "yolo26s-seg.safetensors")
    )
    _default_yolo = "yolo26s-seg.safetensors" if _has_mlx_model else "yolo11s-seg.pt"
    p.add_argument("--yolo-model", type=str, default=_default_yolo,
                   help="YOLO recommandé: yolo26s-seg.safetensors (MLX/Metal, Apple Silicon) "
                        "ou yolo11s-seg.pt (PyTorch, CUDA/CPU).")
    p.add_argument("--dino-model", type=str, default="facebook/dinov2-small",
                   help="DINOv2 small = parfait pour 1660 Ti. -base/-large = trop lourd.")
    # --- Re-ID ---
    p.add_argument("--threshold", type=float, default=0.65,
                   help="Seuil cosine Re-ID normal (0.6-0.75).")
    p.add_argument("--loop-threshold", type=float, default=0.45,
                   help="Seuil permissif juste après un rebobinage vidéo "
                        "(ré-id cross-loop).")
    p.add_argument("--loop-grace-frames", type=int, default=60,
                   help="Frames après rebobinage pendant lesquelles --loop-threshold s'applique.")
    p.add_argument("--max-updates", type=int, default=30,
                   help="Stoppe la mise à jour EMA de chaque bovin après N updates. "
                        "Empêche la dérive de l'embedding de référence.")
    p.add_argument("--db", type=str, default="cattle_db.pkl")
    # --- YOLO runtime ---
    p.add_argument("--conf", type=float, default=0.4,
                   help="Confiance min YOLO. 0.4 = bon plein air.")
    p.add_argument("--device", type=str, default="auto",
                   help="'auto' (CUDA si dispo), 'cpu', 'cuda:0'.")
    p.add_argument("--mlx", action="store_true",
                   help="Utiliser YOLO26 MLX (Metal GPU Apple) au lieu de YOLO11 "
                        "PyTorch MPS. ~2.6× plus rapide sur M1/M2/M3/M4.")
    p.add_argument("--skip-frames", type=int, default=0,
                   help="0 = toutes les frames. 1 = 1 sur 2 (~2x FPS).")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Taille d'inférence YOLO. 640=équilibre. "
                        "416=max FPS. 800=+précision.")
    p.add_argument("--embed-every", type=int, default=10,
                   help="Recalculer l'embedding DINOv2 tous les N frames (perf).")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


# Flask
app = Flask(__name__, template_folder="templates", static_folder="static")
app.json = NumpyJSONProvider(app)


@app.route("/")
def index():
    """Point d'entrée du worker API. L'UI est servie par Bun (port 8000)."""
    return jsonify({
        "service": "boeuf-tracker-worker",
        "version": "2.0",
        "endpoints": ["/api/stats", "/api/devices", "/api/settings",
                      "/api/videos", "/api/breeds", "/video_feed"],
        "ui": "L'interface web est servie par Bun sur http://localhost:8000",
    })


@app.route("/video_feed")
def video_feed():
    """Snapshot JPEG unique (polling côté client) — évite le timeout 524 de Cloudflare
    sur les streams MJPEG keep-alive longs."""
    with STATE["frame_lock"]:
        jpg = STATE["frame_jpg"]
    if jpg is None:
        # Placeholder 1x1 transparent : évite 204 (mal géré par certains navigateurs)
        # tant que le processor n'a pas encore produit sa première frame.
        placeholder = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
            b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
            b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
            b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04"
            b"\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q"
            b"\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17"
            b"\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz"
            b"\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99"
            b"\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7"
            b"\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5"
            b"\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1"
            b"\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
            b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd0\xff\xd9"
        )
        resp = Response(placeholder, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp
    resp = Response(jpg, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/stats")
def stats():
    return jsonify({
        "fps": round(STATE["fps"], 1),
        "device": STATE["device"],
        "started_at": STATE["started_at"],
        "frame_count": STATE["frame_count"],
        "source": STATE["source"],
        "source_label": STATE["source_label"],
        "current_source_path": STATE["current_source_path"],
        "active": STATE["active_animals"],
        "events": STATE["events"],
        "behavior": STATE["behavior"],
        "current": {
            "yolo_model": STATE.get("yolo_model_current"),
            "imgsz": STATE.get("imgsz_current"),
            "embed_every": STATE.get("embed_every_current"),
            "threshold": STATE.get("threshold_current"),
            "conf": STATE.get("conf_current"),
        },
        "desired": {
            "yolo_model": STATE.get("desired_yolo_model"),
            "imgsz": STATE.get("desired_imgsz"),
            "embed_every": STATE.get("desired_embed_every"),
            "threshold": STATE.get("desired_threshold"),
            "conf": STATE.get("desired_conf"),
        },
    })


def _mps_available() -> bool:
    """Vérifie si MPS (Apple Silicon PyTorch) est disponible."""
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


@app.route("/api/devices")
def list_devices():
    import torch
    available = []
    gpus = []
    # MLX sur Apple Silicon (Metal GPU) — priorité maximale
    if STATE["device"] == "mlx" or _mlx_available():
        available.append("mlx")
        gpus.append({"index": 0, "id": "mlx", "name": "Apple Metal GPU (MLX)"})
    # CUDA
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus.append({
                "index": i,
                "id": f"cuda:{i}",
                "name": torch.cuda.get_device_name(i),
            })
            available.append(f"cuda:{i}")
    # MPS (fallback Apple)
    if _mps_available():
        available.append("mps")
    # CPU en dernier recours uniquement
    if not available:
        available.append("cpu")
    return jsonify({
        "available": available,
        "current": STATE["device"],
        "gpus": gpus,
    })


@app.route("/api/device", methods=["POST"])
def set_device():
    data = request.get_json(silent=True) or {}
    requested = (data.get("device") or "auto").strip().lower()
    resolved = resolve_device(requested)
    if requested.startswith("cuda") and resolved == "cpu":
        return jsonify({"ok": False, "error": "CUDA non disponible"}), 400
    if resolved == STATE["device"]:
        return jsonify({"ok": True, "device": resolved, "changed": False})
    STATE["desired_device"] = resolved
    return jsonify({"ok": True, "device": resolved, "changed": True})


@app.route("/api/upload-video", methods=["POST"])
def upload_video():
    import time
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "aucun fichier"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"ok": False, "error": "nom vide"}), 400
    name = secure_filename(f.filename)
    if not name:
        return jsonify({"ok": False, "error": "nom invalide"}), 400
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}
    ext = os.path.splitext(name)[1].lower()
    if ext not in allowed:
        return jsonify({"ok": False, "error": f"extension non supportée: {ext}"}), 400
    ts_name = f"{int(time.time())}_{name}"
    dest = os.path.join(UPLOAD_DIR, ts_name)
    f.save(dest)
    STATE["desired_source"] = dest
    return jsonify({"ok": True, "filename": name, "path": dest})


@app.route("/api/source/webcam", methods=["POST"])
def source_webcam():
    data = request.get_json(silent=True) or {}
    idx = int(data.get("index", 0))
    STATE["desired_source"] = idx
    return jsonify({"ok": True, "index": idx})


@app.route("/api/videos")
def list_videos():
    """Liste les vidéos présentes dans le dossier projet (racine + uploads/)."""
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}
    project_root = os.path.dirname(os.path.abspath(__file__))
    found = []
    # Racine du projet (exclure uploads/ pour éviter doublons)
    for entry in os.listdir(project_root):
        full = os.path.join(project_root, entry)
        if os.path.isfile(full) and os.path.splitext(entry)[1].lower() in video_exts:
            found.append({
                "name": entry,
                "path": os.path.abspath(full),
                "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
                "source": "project",
            })
    # Dossier uploads/
    if os.path.isdir(UPLOAD_DIR):
        for entry in os.listdir(UPLOAD_DIR):
            full = os.path.join(UPLOAD_DIR, entry)
            if os.path.isfile(full) and os.path.splitext(entry)[1].lower() in video_exts:
                found.append({
                    "name": entry,
                    "path": os.path.abspath(full),
                    "size_mb": round(os.path.getsize(full) / 1024 / 1024, 1),
                    "source": "uploads",
                })
    return jsonify({"videos": found})


@app.route("/api/source/file", methods=["POST"])
def source_file():
    """Bascule vers un fichier vidéo par son chemin."""
    data = request.get_json(silent=True) or {}
    path = data.get("path", "").strip()
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": "fichier introuvable"}), 400
    STATE["desired_source"] = path
    return jsonify({"ok": True, "path": path})


@app.route("/api/diag")
def diag():
    return jsonify({
        "state": {
            "device": STATE["device"],
            "source": STATE["source"],
            "source_label": STATE["source_label"],
            "current_source_path": STATE["current_source_path"],
            "desired_source": STATE["desired_source"],
            "fps": round(STATE["fps"], 1),
            "frame_count": STATE["frame_count"],
            "active_count": len(STATE["active_animals"]),
            "events_count": len(STATE["events"]),
            "has_frame": STATE["frame_jpg"] is not None,
        },
        "uploads": os.listdir(UPLOAD_DIR) if os.path.isdir(UPLOAD_DIR) else [],
    })


@app.route("/api/db/reset", methods=["POST"])
def reset_db():
    """Purge la base de données des embeddings."""
    from database import EmbeddingDatabase
    data = request.get_json(silent=True) or {}
    db_path = data.get("path", "cattle_db.pkl")
    if os.path.exists(db_path):
        os.remove(db_path)
    # Aussi purger l'in-memory
    STATE["events"].insert(0, "DB RESET")
    STATE["events"] = STATE["events"][:30]
    return jsonify({"ok": True, "message": "base purgée", "path": db_path})


@app.route("/api/rematch", methods=["POST"])
def rematch():
    """
    Force le re-appariement des tracks en cours en vidant le cache track_id_to_name.
    Utile apres un changement de --threshold pour re-evaluer les identites.
    Le prochain passage de chaque bovin re-passera par db.match() avec
    le seuil courant (ou --loop-threshold si on vient de boucler).
    """
    STATE["desired_rematch"] = True
    STATE["events"].insert(0, "REMATCH demande")
    STATE["events"] = STATE["events"][:30]
    return jsonify({"ok": True, "message": "re-appariement demande"})


@app.route("/api/restart", methods=["POST"])
def restart_server():
    """
    Demande un redémarrage du serveur (utilisé par le watcher).
    Le watcher détecte et relance proprement.
    """
    STATE["events"].insert(0, "RESTART demandé")
    STATE["events"] = STATE["events"][:30]
    # Force un crash volontaire après quelques frames pour que le watcher reprenne
    def _kill():
        import os as _os, time as _time
        _time.sleep(2)
        _os._exit(0)
    import threading
    threading.Thread(target=_kill, daemon=True).start()
    return jsonify({"ok": True, "message": "redémarrage dans 2s"})


@app.route("/api/settings", methods=["GET"])
def get_settings():
    """Retourne les paramètres actuels et ceux en attente."""
    return jsonify({
        "current": {
            "yolo_model": STATE["yolo_model_current"],
            "imgsz": STATE["imgsz_current"],
            "embed_every": STATE["embed_every_current"],
            "threshold": STATE["threshold_current"],
            "conf": STATE["conf_current"],
        },
        "models_available": STATE["models_available"],
        "desired": {
            "yolo_model": STATE.get("desired_yolo_model"),
            "imgsz": STATE["desired_imgsz"],
            "embed_every": STATE["desired_embed_every"],
            "threshold": STATE["desired_threshold"],
            "conf": STATE["desired_conf"],
        },
    })


@app.route("/api/breeds")
def list_breeds():
    """Liste toutes les races connues avec leurs descriptions (pour l'UI Bun)."""
    from breed import BREEDS
    return jsonify({
        "breeds": [
            {"name": name, **info}
            for name, info in BREEDS.items()
        ],
        "count": len(BREEDS),
    })


@app.route("/api/breeds/<path:breed_name>")
def breed_detail(breed_name):
    """Détail d'une race spécifique (origine, robe, caractéristiques)."""
    from breed import get_breed_info
    info = get_breed_info(breed_name)
    if info is None:
        return jsonify({"ok": False, "error": f"race inconnue: {breed_name}"}), 404
    return jsonify({"name": breed_name, **info})


@app.route("/api/animals")
def list_animals():
    """Liste tous les animaux identifiés en DB (avec leur race + nom propre)."""
    from database import EmbeddingDatabase
    from names import make_name_generator
    db = EmbeddingDatabase(path="cattle_db.pkl")
    name_gen = make_name_generator(db)
    animals = []
    for name, data in db.animals.items():
        animals.append({
            "key": name,                     # Boeuf_001 (interne)
            "name": name_gen.get(name),      # "Marguerite" (nom propre)
            "breed": data.get("breed", "Indeterminee"),
            "breed_confidence": data.get("breed_confidence", 0),
            "coat_swatch": data.get("coat_swatch", "#555555"),
            "breeds_compat": data.get("breeds_compat", []),
            "count": data.get("count", 0),
            "first_seen": data.get("first_seen"),
        })
    return jsonify({
        "animals": animals,
        "count": len(animals),
        "name_mapping": name_gen.all(),  # {Boeuf_001: Marguerite, ...}
    })


@app.route("/api/bench", methods=["GET"])
def bench_fps():
    """
    Benchmark comparatif YOLO sur Apple Silicon.
    Mesure FPS pour:
    - YOLO11s-seg sur MPS (actuel)
    - YOLO26s-seg sur MLX (Apple Metal GPU)
    - DINOv2-small sur MPS (actuel)
    """
    import time, torch, cv2, numpy as np
    from PIL import Image

    results = {
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "yolo_mlx_available": False,
        "yolo_mlx_fps": None,
        "yolo_mlx_model": None,
        "yolo_mlx_error": None,
        "yolo11s_mps_fps": None,
        "yolo11s_mps_error": None,
        "dino_mps_fps": None,
        "dino_mps_error": None,
        "coreml_dino_available": False,
        "coreml_dino_fps": None,
        "coreml_dino_error": None,
    }

    # --- Test image (1 frame, même condition pour tous) ---
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    # Dessine 3 "bovins" simulés pour avoir des détections
    for i in range(3):
        x, y = 100 + i * 180, 150
        cv2.ellipse(dummy_img, (x, y), (60, 40), 0, 0, 360, (255, 255, 255), -1)

    warmup_runs = 3
    bench_runs = 20

    # ===== 1. YOLO11s-seg sur MPS =====
    if torch.backends.mps.is_available():
        try:
            from ultralytics import YOLO
            model11 = YOLO("yolo11s-seg.pt")
            model11.to("mps")

            # Warmup
            for _ in range(warmup_runs):
                model11.predict(source=dummy_img, device="mps")

            # Bench
            t0 = time.perf_counter()
            for _ in range(bench_runs):
                model11.predict(source=dummy_img, device="mps")
            elapsed = time.perf_counter() - t0
            results["yolo11s_mps_fps"] = round(bench_runs / elapsed, 1)
        except Exception as e:
            results["yolo11s_mps_error"] = str(e)

    # ===== 2. YOLO26s-seg sur MLX =====
    try:
        import os as _os
        from yolo26mlx import YOLO as YOLO26

        # Use pre-converted safetensors if available, otherwise .pt (will convert on first run)
        pt_path = _os.path.join(_os.path.dirname(__file__), "yolo26s-seg.pt")
        safetensors_path = _os.path.join(_os.path.dirname(__file__), "yolo26s-seg.safetensors")
        model_path = safetensors_path if _os.path.exists(safetensors_path) else pt_path

        model26 = YOLO26(model_path)
        results["yolo_mlx_available"] = True

        # Warmup
        for _ in range(warmup_runs):
            model26.predict(dummy_img)

        # Bench
        t0 = time.perf_counter()
        for _ in range(bench_runs):
            model26.predict(dummy_img)
        elapsed = time.perf_counter() - t0
        results["yolo_mlx_fps"] = round(bench_runs / elapsed, 1)
        results["yolo_mlx_model"] = "yolo26s-seg"
    except ImportError:
        results["yolo_mlx_error"] = "yolo26mlx non installé"
    except Exception as e:
        results["yolo_mlx_error"] = str(e)

    # ===== 3. DINOv2-small sur MPS (1 crop) =====
    if torch.backends.mps.is_available():
        try:
            from transformers import AutoModel, AutoImageProcessor

            dino_model = AutoModel.from_pretrained("facebook/dinov2-small")
            dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
            dino_model.to("mps")
            dino_model.eval()

            crop = cv2.resize(dummy_img[100:350, 80:200], (224, 224))
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(crop_rgb)

            # Warmup
            for _ in range(warmup_runs):
                with torch.no_grad():
                    inputs = dino_proc(images=pil, return_tensors="pt").to("mps")
                    dino_model(**inputs)

            # Bench
            t0 = time.perf_counter()
            for _ in range(bench_runs):
                with torch.no_grad():
                    inputs = dino_proc(images=pil, return_tensors="pt").to("mps")
                    dino_model(**inputs)
            elapsed = time.perf_counter() - t0
            results["dino_mps_fps"] = round(bench_runs / elapsed, 1)
        except Exception as e:
            results["dino_mps_error"] = str(e)

    # ===== 4. DINOv2 sur CoreML (si déjà converti) =====
    import os as _os
    coreml_path = _os.path.join(_os.path.dirname(__file__), "dinov2-small.mlpackage")
    if _os.path.isdir(coreml_path):
        try:
            import coremltools as ct
            coreml_model = ct.models.MLModel(coreml_path)
            results["coreml_dino_available"] = True

            # Warmup
            for _ in range(warmup_runs):
                coreml_model.predict({"input_image": pil})

            # Bench
            t0 = time.perf_counter()
            for _ in range(bench_runs):
                coreml_model.predict({"input_image": pil})
            elapsed = time.perf_counter() - t0
            results["coreml_dino_fps"] = round(bench_runs / elapsed, 1)
        except Exception as e:
            results["coreml_dino_error"] = str(e)

    return jsonify(results)


@app.route("/api/settings", methods=["POST"])
def set_settings():
    """Change les paramètres à chaud (sans redémarrer).

    Body JSON, n'importe quelle combinaison de:
    - yolo_model (str): chemin ou nom du modèle YOLO
    - imgsz (int): résolution d'inférence YOLO (320, 416, 640, 960, 1280)
    - embed_every (int): fréquence re-embedding DINOv2
    - threshold (float): seuil cosine Re-ID (0-1)
    - conf (float): seuil confiance YOLO (0-1)
    """
    data = request.get_json(silent=True) or {}
    accepted = []
    if "yolo_model" in data:
        STATE["desired_yolo_model"] = str(data["yolo_model"])
        accepted.append(f"yolo_model={data['yolo_model']}")
    if "imgsz" in data:
        v = int(data["imgsz"])
        if v not in (320, 416, 512, 640, 800, 960, 1280):
            return jsonify({"ok": False, "error": f"imgsz invalide: {v}"}), 400
        STATE["desired_imgsz"] = v
        accepted.append(f"imgsz={v}")
    if "embed_every" in data:
        v = max(1, int(data["embed_every"]))
        STATE["desired_embed_every"] = v
        accepted.append(f"embed_every={v}")
    if "threshold" in data:
        v = float(data["threshold"])
        if not 0 <= v <= 1:
            return jsonify({"ok": False, "error": "threshold doit être entre 0 et 1"}), 400
        STATE["desired_threshold"] = v
        accepted.append(f"threshold={v}")
    if "conf" in data:
        v = float(data["conf"])
        if not 0 < v <= 1:
            return jsonify({"ok": False, "error": "conf doit être entre 0 et 1"}), 400
        STATE["desired_conf"] = v
        accepted.append(f"conf={v}")
    if accepted:
        STATE["events"].insert(0, "SETTINGS " + " ".join(accepted))
        STATE["events"] = STATE["events"][:30]
    return jsonify({"ok": True, "accepted": accepted})


def main():
    args = parse_args()
    src_display = (
        f"webcam ({args.source})" if str(args.source).isdigit()
        else os.path.basename(args.source)
    )
    log_banner(
        "BOEUF TRACKER — Interface web",
        [
            f"URL       : http://{args.host}:{args.port}",
            f"Source    : {src_display}",
            f"YOLO      : {args.yolo_model}",
            f"DINOv2    : {args.dino_model}",
            f"Device    : {args.device}",
            f"Imgsz     : {args.imgsz}",
            f"Confiance : {args.conf}",
            f"Re-ID     : thr={args.threshold}  loop-thr={args.loop_threshold}  "
            f"grace={args.loop_grace_frames}f  max-upd={args.max_updates}",
        ],
    )
    start_detection_thread(args)
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()