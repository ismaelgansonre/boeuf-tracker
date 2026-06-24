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

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--db", type=str, default="cattle_db.pkl")
    p.add_argument("--yolo-model", type=str, default="yolo11l-seg.pt")
    p.add_argument("--dino-model", type=str, default="facebook/dinov2-small")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--skip-frames", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Taille d'inférence YOLO. 1280 = +precis mais +lent")
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


def main():
    args = parse_args()
    print("=" * 64)
    print("  BOEUF TRACKER — Interface web")
    print(f"  http://{args.host}:{args.port}")
    print(f"  YOLO: {args.yolo_model} | Device: {args.device} | Imgsz: {args.imgsz}")
    print("=" * 64)
    start_detection_thread(args)
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()