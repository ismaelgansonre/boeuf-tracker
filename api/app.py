"""
api/app.py
----------
Flask application with routes for the web interface.
"""
import argparse
import os
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

import sys
sys.path.insert(0, '..')
from config import config, UPLOADS_DIR
from utils.console import banner, info, ok, warn, err
from utils.state import STATE
from core.processor import start_detection_thread, resolve_device, _mlx_available


def _pick_default_source() -> str:
    """Pick default source: first video at project root, else webcam."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    """Parse CLI arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default=_pick_default_source())
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--yolo-model", type=str, default=config.model.yolo_model)
    p.add_argument("--dino-model", type=str, default=config.model.dino_model)
    p.add_argument("--threshold", type=float, default=config.detection.reid_threshold)
    p.add_argument("--loop-threshold", type=float, default=config.detection.loop_threshold)
    p.add_argument("--loop-grace-frames", type=int, default=config.detection.loop_grace_frames)
    p.add_argument("--max-updates", type=int, default=config.detection.max_ema_updates)
    p.add_argument("--db", type=str, default="cattle_db.pkl")
    p.add_argument("--conf", type=float, default=config.detection.conf_threshold)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--mlx", action="store_true")
    return p.parse_args()


def create_app(source: str = None, yolo_model: str = None, dino_model: str = None,
               threshold: float = None, db_path: str = None, conf: float = None,
               device: str = None, mlx: bool = False, loop_threshold: float = 0.55,
               loop_grace_frames: int = 60, max_updates: int = 30) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="web/public")
    app.config["SECRET_KEY"] = "boeuf-tracker-secret"
    app.config["UPLOAD_FOLDER"] = UPLOADS_DIR

    # Apply config
    source = source or _pick_default_source()
    yolo_model = yolo_model or config.model.auto_yolo()
    threshold = threshold or config.detection.reid_threshold
    conf = conf or config.detection.conf_threshold

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(
            _generate_mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.route("/api/stats")
    def stats():
        return jsonify({
            "fps": STATE.get("fps", 0),
            "frame_count": STATE.get("frame_count", 0),
            "active_animals": STATE.get("active_animals", []),
            "device": STATE.get("device", "cpu"),
            "source": STATE.get("source_label", ""),
            "yolo_model": STATE.get("yolo_model_current", ""),
            "events": STATE.get("events", [])[-10:],
        })

    @app.route("/api/upload-video", methods=["POST"])
    def upload_video():
        if "video" not in request.files:
            return jsonify({"error": "No video file"}), 400
        file = request.files["video"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        STATE["desired_source"] = filepath
        return jsonify({"success": True, "path": filepath})

    @app.route("/api/source/webcam", methods=["POST"])
    def source_webcam():
        data = request.get_json() or {}
        source_id = data.get("source", "0")
        STATE["desired_source"] = source_id
        return jsonify({"success": True})

    @app.route("/api/device", methods=["POST"])
    def device():
        data = request.get_json() or {}
        device = data.get("device", "auto")
        STATE["desired_device"] = device
        return jsonify({"success": True, "device": device})

    @app.route("/api/devices")
    def devices():
        from utils.state import STATE
        return jsonify({
            "devices": ["auto", "cuda", "mps", "cpu"],
            "current": STATE.get("device", "cpu"),
        })

    @app.route("/api/diag")
    def diag():
        import torch
        return jsonify({
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "mps_available": torch.backends.mps.is_available(),
            "mlx_available": _mlx_available,
            "models_available": STATE.get("models_available", []),
        })

    @app.route("/api/db/reset", methods=["POST"])
    def db_reset():
        from core.database import EmbeddingDatabase
        db = EmbeddingDatabase(path=db_path or config.model.reid_engine or "cattle_db.pkl")
        db.reset()
        return jsonify({"success": True})

    # Start detection thread
    start_detection_thread(
        source=source,
        yolo_model=yolo_model,
        dino_model=dino_model,
        threshold=threshold,
        db_path=db_path,
        conf=conf,
        device=device,
        mlx=mlx,
        loop_threshold=loop_threshold,
        loop_grace_frames=loop_grace_frames,
        max_updates=max_updates,
    )

    return app


def _generate_mjpeg():
    """Generator for MJPEG stream."""
    while True:
        with STATE["frame_lock"]:
            frame_jpg = STATE.get("frame_jpg")
        if frame_jpg:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_jpg + b"\r\n")
        else:
            import time
            time.sleep(0.1)


def run_app():
    """Run the Flask app."""
    args = parse_args()
    app = create_app(
        source=args.source,
        yolo_model=args.yolo_model,
        dino_model=args.dino_model,
        threshold=args.threshold,
        db_path=args.db,
        conf=args.conf,
        device=args.device,
        mlx=args.mlx,
        loop_threshold=args.loop_threshold,
        loop_grace_frames=args.loop_grace_frames,
        max_updates=args.max_updates,
    )
    banner("Boeuf Tracker", [
        f"Source: {args.source}",
        f"YOLO: {args.yolo_model}",
        f"Device: {args.device}",
        f"Server: http://{args.host}:{args.port}",
    ])
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    run_app()
