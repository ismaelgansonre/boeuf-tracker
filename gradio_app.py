"""
gradio_app.py
-------------
UI Gradio EXPÉRIMENTALE pour Boeuf Tracker.

⚠️  Le tunnel officiel `gradio.live` (share=True) a un bug upstream
    (gradio-app/gradio#11553) qui casse les versions Gradio ≥ 5.36.2
    avec une erreur 404 sur manifest.json. Pour l'instant on utilise
    Flask + ngrok (voir colab.ipynb).

Si tu veux quand même essayer Gradio avec ton propre tunnel ngrok :
    python gradio_app.py --no-share --port 7860
    # puis dans un autre terminal :
    ngrok http 7860

Démarre le processor dans un thread daemon et sert l'UI via Gradio.
"""
import argparse
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

# --- Gradio doit être installé avant le reste ---
try:
    import gradio as gr
    print(f"[gradio] version {gr.__version__}", flush=True)
except ImportError:
    print("[gradio] FATAL: gradio non installé. Fais: pip install gradio>=4.20.0",
          file=sys.stderr, flush=True)
    sys.exit(1)

from state import STATE
from processor import start_detection_thread


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", dest="share", action="store_true", default=True)
    p.add_argument("--no-share", dest="share", action="store_false")
    p.add_argument("--yolo-model", type=str, default="yolo11s-seg.pt")
    p.add_argument("--dino-model", type=str, default="facebook/dinov2-small")
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--loop-threshold", type=float, default=0.45)
    p.add_argument("--loop-grace-frames", type=int, default=60)
    p.add_argument("--max-updates", type=int, default=30)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--embed-every", type=int, default=10)
    p.add_argument("--db", type=str, default="cattle_db.pkl")
    p.add_argument("--skip-frames", type=int, default=0)
    return p.parse_args()


# ---------- Callbacks sûrs (ne lèvent jamais) ----------
def safe_get_frame():
    try:
        with STATE["frame_lock"]:
            jpg = STATE["frame_jpg"]
        if jpg is None:
            return None
        arr = np.frombuffer(jpg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"[gradio] get_frame: {e}", file=sys.stderr, flush=True)
        return None


def safe_get_stats():
    try:
        return {
            "fps": round(float(STATE.get("fps", 0.0)), 1),
            "frames": int(STATE.get("frame_count", 0)),
            "device": str(STATE.get("device", "?")),
            "source": str(STATE.get("source_label", "?")),
            "active": [a.get("name") for a in STATE.get("active_animals", [])],
            "events": (STATE.get("events") or [])[:5],
            "has_frame": STATE.get("frame_jpg") is not None,
            "yolo": STATE.get("yolo_model_current", "?"),
            "imgsz": STATE.get("imgsz_current", "?"),
        }
    except Exception as e:
        print(f"[gradio] get_stats: {e}", file=sys.stderr, flush=True)
        return {}


def list_videos() -> list[str]:
    out: list[str] = []
    try:
        for entry in sorted(Path(".").iterdir()):
            if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
                out.append(str(entry.resolve()))
        up = Path("uploads")
        if up.is_dir():
            for entry in sorted(up.iterdir()):
                if entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
                    out.append(str(entry.resolve()))
    except Exception as e:
        print(f"[gradio] list_videos: {e}", file=sys.stderr, flush=True)
    return out


# ---------- Actions (mutent le STATE) ----------
def action_switch_source(path: str):
    if path and Path(path).exists():
        STATE["desired_source"] = path
        return f"✅ Switch demandé → {Path(path).name}"
    return f"❌ Fichier introuvable : {path}"


def action_switch_webcam():
    STATE["desired_source"] = 0
    return "✅ Switch demandé → webcam (index 0)"


def action_upload_video(file_obj):
    """Accepte un fichier uploadé (objet tempfile gradio) et le déplace dans uploads/."""
    if file_obj is None:
        return "❌ Aucun fichier"
    src = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
    if not src.exists() or src.suffix.lower() not in VIDEO_EXTS:
        return f"❌ Extension non supportée : {src.suffix}"
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    dest = upload_dir / f"{int(time.time())}_{src.name}"
    shutil.copy2(src, dest)
    STATE["desired_source"] = str(dest.resolve())
    return f"✅ Uploadé + switch demandé → {dest.name}"


def action_set_threshold(v):
    try: STATE["desired_threshold"] = float(v)
    except Exception: pass

def action_set_conf(v):
    try: STATE["desired_conf"] = float(v)
    except Exception: pass

def action_set_embed_every(v):
    try: STATE["desired_embed_every"] = int(v)
    except Exception: pass

def action_set_imgsz(v):
    try: STATE["desired_imgsz"] = int(v)
    except Exception: pass

def action_set_model(name):
    if name: STATE["desired_yolo_model"] = name

def action_set_device(dev):
    if dev: STATE["desired_device"] = dev

def action_rematch():
    STATE["desired_rematch"] = True
    STATE["events"].insert(0, "REMATCH demandé via UI")
    STATE["events"] = STATE["events"][:30]
    return "✅ Re-id forcée"

def action_reset_db():
    try:
        db_path = Path(STATE.get("db_path", "cattle_db.pkl"))
        if db_path.exists():
            db_path.unlink()
        STATE["active_animals"] = []
        STATE["track_history"].clear()
        STATE["events"].insert(0, "DB RESET via UI")
        STATE["events"] = STATE["events"][:30]
        return f"✅ Base purgée ({db_path})"
    except Exception as e:
        return f"❌ {e}"


# ---------- Construction de l'UI ----------
def build_ui():
    with gr.Blocks(title="Boeuf Tracker", theme=gr.themes.Default()) as demo:
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
                status = gr.Markdown("⏳ En attente de la première frame…")

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
                threshold_sl = gr.Slider(
                    0.30, 0.95,
                    value=STATE.get("threshold_current", 0.65),
                    step=0.01, label="Seuil Re-ID",
                )
                conf_sl = gr.Slider(
                    0.10, 0.90,
                    value=STATE.get("conf_current", 0.40),
                    step=0.05, label="Confiance YOLO",
                )
                embed_sl = gr.Slider(
                    1, 60,
                    value=STATE.get("embed_every_current", 10),
                    step=1, label="Re-embed tous les N frames",
                )

                with gr.Row():
                    btn_rematch = gr.Button("🔄 Re-match")
                    btn_reset = gr.Button("🗑 Reset DB", variant="stop")

                stats_json = gr.JSON(label="Stats live", value={})

        # === Auto-refresh via Timer (2 fps cam, 1 Hz stats) ===
        cam_timer = gr.Timer(0.5, active=True)
        cam_timer.tick(safe_get_frame, outputs=cam)

        stats_timer = gr.Timer(1.0, active=True)
        stats_timer.tick(safe_get_stats, outputs=stats_json)

        # === Wiring des contrôles ===
        btn_load.click(action_switch_source, inputs=[video_dd], outputs=[status])
        btn_webcam.click(action_switch_webcam, outputs=[status])
        model_dd.change(action_set_model, inputs=[model_dd])
        device_dd.change(action_set_device, inputs=[device_dd])
        imgsz_dd.change(action_set_imgsz, inputs=[imgsz_dd])
        threshold_sl.release(action_set_threshold, inputs=[threshold_sl])
        conf_sl.release(action_set_conf, inputs=[conf_sl])
        embed_sl.release(action_set_embed_every, inputs=[embed_sl])
        btn_rematch.click(action_rematch, outputs=[status])
        btn_reset.click(action_reset_db, outputs=[status])

    return demo


# ---------- Main ----------
def main():
    args = parse_args()
    STATE["db_path"] = args.db

    print(f"[gradio] Démarrage du thread processor…", flush=True)
    try:
        start_detection_thread(args)
        print(f"[gradio] Processor démarré.", flush=True)
    except Exception as e:
        print(f"[gradio] FATAL start_detection_thread: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    print(f"[gradio] Construction de l'UI…", flush=True)
    try:
        demo = build_ui()
        print(f"[gradio] UI construite.", flush=True)
    except Exception as e:
        print(f"[gradio] FATAL build_ui: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    print(f"[gradio] Lancement sur {args.host}:{args.port} (share={args.share})…", flush=True)
    try:
        demo.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True,
            prevent_thread_lock=False,
        )
    except KeyboardInterrupt:
        print(f"[gradio] Arrêté par l'utilisateur.", flush=True)
    except Exception as e:
        print(f"[gradio] FATAL launch: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
