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
from processor import start_detection_thread, resolve_device
from console import banner as log_banner, info, ok, warn, err

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def parse_args():
    """
    Défauts optimisés pour GTX 1660 Ti (6 GB) — équilibre perf / détection.
    Lance simplement:    python app.py
    Override possible via CLI (voir PERF_GUIDE_1660TI.md).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0",
                   help="'0' = webcam | chemin vidéo | URL RTSP")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    # --- Modèles (recommandés 1660 Ti) ---
    p.add_argument("--yolo-model", type=str, default="yolo11s-seg.pt",
                   help="YOLO recommandé: yolo11s-seg.pt (GTX 1660 Ti). "
                        "Évite l/x-seg sur 6GB VRAM.")
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
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def gen():
        boundary = b"--frame"
        while True:
            with STATE["frame_lock"]:
                jpg = STATE["frame_jpg"]
            if jpg is None:
                import time
                time.sleep(0.05)
                continue
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            import time
            time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


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


@app.route("/api/devices")
def list_devices():
    import torch
    available = ["cpu"]
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus.append({
                "index": i,
                "id": f"cuda:{i}",
                "name": torch.cuda.get_device_name(i),
            })
            available.append(f"cuda:{i}")
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
    log_banner(
        "BOEUF TRACKER — Interface web",
        [
            f"URL       : http://{args.host}:{args.port}",
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