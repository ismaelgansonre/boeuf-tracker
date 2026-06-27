"""
gradio_app.py
-------------
UI Gradio pour Boeuf Tracker.

Lance le processor (YOLO + DINOv2) dans un thread daemon et sert l'UI
via Gradio avec `share=True` (tunnel *.gradio.live gratuit, sans config).

Avantages vs Flask + cloudflared :
  - Pas de timeout 524 sur les streams longs (Gradio utilise WebSocket).
  - Pas besoin de configurer un tunnel manuellement.
  - UI responsive avec composants interactifs natifs.
  - Compatible avec Flask en parallèle (même STATE partagé) pour le monitoring.

Usage local :
    python gradio_app.py

Usage notebook Colab (voir colab.ipynb) :
    subprocess.Popen([sys.executable, "gradio_app.py", "--source", "...", ...])
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from state import STATE
from processor import start_detection_thread


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", default=True,
                   help="Crée un tunnel gradio.live public (défaut: True)")
    p.add_argument("--no-share", dest="share", action="store_false")
    p.add_argument("--yolo-model", type=str, default="yolo11s-seg.pt")
    p.add_argument("--dino-model", type=str, default="facebook/dinov2-small")
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--embed-every", type=int, default=10)
    p.add_argument("--db", type=str, default="cattle_db.pkl")
    p.add_argument("--skip-frames", type=int, default=0)
    return p.parse_args()


# ---------- Lecture du dernier frame depuis le STATE partagé ----------
_last_jpg_signature: tuple | None = None


def get_latest_frame() -> np.ndarray | None:
    """Retourne le dernier frame en RGB numpy, ou None si pas encore prêt."""
    global _last_jpg_signature
    with STATE["frame_lock"]:
        jpg = STATE["frame_jpg"]
    if jpg is None:
        return None
    sig = (id(jpg), len(jpg))
    if sig == _last_jpg_signature:
        return None
    _last_jpg_signature = sig
    arr = np.frombuffer(jpg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def get_stats_dict() -> dict:
    """Petit résumé pour affichage live (compatible avec ce que /api/stats retourne)."""
    return {
        "fps": round(STATE.get("fps", 0.0), 1),
        "frame_count": STATE.get("frame_count", 0),
        "device": STATE.get("device", "?"),
        "source": STATE.get("source_label", "?"),
        "active": [a.get("name") for a in STATE.get("active_animals", [])],
        "events": (STATE.get("events") or [])[:5],
        "current": {
            "yolo_model": STATE.get("yolo_model_current", "?"),
            "imgsz": STATE.get("imgsz_current", "?"),
            "threshold": STATE.get("threshold_current", "?"),
            "conf": STATE.get("conf_current", "?"),
            "embed_every": STATE.get("embed_every_current", "?"),
        },
        "has_frame": STATE.get("frame_jpg") is not None,
    }


# ---------- Helpers d'action (mutent le STATE) ----------
def action_switch_source(path: str):
    if path and Path(path).exists():
        STATE["desired_source"] = path
        return f"✅ Switch demandé → {Path(path).name}"
    return f"❌ Fichier introuvable : {path}"


def action_switch_webcam():
    STATE["desired_source"] = 0
    return "✅ Switch demandé → webcam (index 0)"


def action_upload_video(file_path: str | None):
    if not file_path:
        return "❌ Aucun fichier", gr.update()
    src = Path(file_path)
    if src.suffix.lower() not in VIDEO_EXTS:
        return f"❌ Extension non supportée : {src.suffix}", gr.update()
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    dest = upload_dir / f"{int(time.time())}_{src.name}"
    shutil.copy2(src, dest)
    STATE["desired_source"] = str(dest.resolve())
    return f"✅ Uploadé + switch demandé → {dest.name}", gr.update(choices=list_videos())


def action_set_threshold(v: float):
    STATE["desired_threshold"] = float(v)


def action_set_conf(v: float):
    STATE["desired_conf"] = float(v)


def action_set_embed_every(v: int):
    STATE["desired_embed_every"] = int(v)


def action_set_imgsz(v: int):
    STATE["desired_imgsz"] = int(v)


def action_set_model(name: str):
    if name:
        STATE["desired_yolo_model"] = name


def action_set_device(dev: str):
    if dev:
        STATE["desired_device"] = dev


def action_rematch():
    STATE["desired_rematch"] = True
    STATE["events"].insert(0, "REMATCH demandé via UI")
    STATE["events"] = STATE["events"][:30]
    return "✅ Re-id forcée — prochaines frames"


def action_reset_db():
    db_path = Path(STATE.get("db_path", "cattle_db.pkl"))
    if db_path.exists():
        db_path.unlink()
    STATE["active_animals"] = []
    STATE["track_history"].clear()
    STATE["events"].insert(0, "DB RESET via UI")
    STATE["events"] = STATE["events"][:30]
    return f"✅ Base purgée ({db_path})"


# ---------- Liste des vidéos dans le dossier projet ----------
def list_videos() -> list[str]:
    out: list[str] = []
    for entry in sorted(Path(".").iterdir()):
        if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
            out.append(str(entry.resolve()))
    uploads = Path("uploads")
    if uploads.is_dir():
        for entry in sorted(uploads.iterdir()):
            if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
                out.append(str(entry.resolve()))
    return out


# ---------- Construction de l'UI ----------
def build_ui():
    import gradio as gr

    with gr.Blocks(title="Boeuf Tracker", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🐄 Boeuf Tracker\nYOLOv11 + DINOv2 — Re-ID temps réel")

        with gr.Row():
            with gr.Column(scale=3):
                cam = gr.Image(
                    label="Flux temps réel",
                    type="numpy",
                    height=540,
                    interactive=False,
                    show_label=False,
                )
                status = gr.Markdown("⏳ Démarrage du processor…")

            with gr.Column(scale=1):
                gr.Markdown("### 📂 Source")
                video_dd = gr.Dropdown(
                    choices=list_videos(),
                    label="Vidéos du projet",
                    type="value",
                )
                with gr.Row():
                    btn_load = gr.Button("▶ Charger", variant="primary")
                    btn_webcam = gr.Button("📷 Webcam")
                upload = gr.File(
                    label="Upload vidéo (.mp4, .mov, …)",
                    file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
                )
                upload_status = gr.Markdown("")

                gr.Markdown("### ⚙️ Modèle & device")
                model_dd = gr.Dropdown(
                    choices=STATE.get("models_available", []),
                    value=STATE.get("yolo_model_current"),
                    label="Modèle YOLO",
                )
                device_dd = gr.Dropdown(
                    choices=["auto", "cpu", "cuda:0"],
                    value="auto",
                    label="Device",
                )
                imgsz_dd = gr.Radio(
                    choices=[320, 416, 640, 960, 1280],
                    value=STATE.get("imgsz_current", 640),
                    label="imgsz",
                )

                gr.Markdown("### 🎚️ Seuils")
                threshold_sl = gr.Slider(0.30, 0.95, value=STATE.get("threshold_current", 0.65),
                                          step=0.01, label="Seuil Re-ID")
                conf_sl = gr.Slider(0.10, 0.90, value=STATE.get("conf_current", 0.40),
                                     step=0.05, label="Confiance YOLO")
                embed_sl = gr.Slider(1, 60, value=STATE.get("embed_every_current", 10),
                                      step=1, label="Re-embed tous les N frames")

                with gr.Row():
                    btn_rematch = gr.Button("🔄 Re-match")
                    btn_reset = gr.Button("🗑 Reset DB", variant="stop")

                stats_json = gr.JSON(label="Stats live", value={})

        # === Wiring ===
        timer_cam = gr.Timer(0.04)        # ~25 fps
        timer_stats = gr.Timer(1.0)        # 1 Hz

        timer_cam.tick(get_latest_frame, outputs=cam)
        timer_stats.tick(get_stats_dict, outputs=stats_json)

        # Source
        btn_load.click(action_switch_source, inputs=[video_dd], outputs=[status])
        btn_webcam.click(action_switch_webcam, outputs=[status])
        upload.upload(action_upload_video, inputs=[upload], outputs=[upload_status, video_dd])

        # Modèle
        model_dd.change(action_set_model, inputs=[model_dd], outputs=[])
        device_dd.change(action_set_device, inputs=[device_dd], outputs=[])
        imgsz_dd.change(action_set_imgsz, inputs=[imgsz_dd], outputs=[])

        # Seuils
        threshold_sl.release(action_set_threshold, inputs=[threshold_sl], outputs=[])
        conf_sl.release(action_set_conf, inputs=[conf_sl], outputs=[])
        embed_sl.release(action_set_embed_every, inputs=[embed_sl], outputs=[])

        # Actions
        btn_rematch.click(action_rematch, outputs=[status])
        btn_reset.click(action_reset_db, outputs=[status])

    return demo


# ---------- Main ----------
def main():
    args = parse_args()

    # Persistance du chemin DB pour le reset
    STATE["db_path"] = args.db

    # Démarre le thread processor (cœur métier — déjà thread-safe via STATE)
    start_detection_thread(args)

    # Construit et lance l'UI
    demo = build_ui()
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        prevent_thread_lock=False,
    )


if __name__ == "__main__":
    main()
