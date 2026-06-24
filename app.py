"""
app.py
------
Serveur Flask minimal qui exécute la détection dans un thread et stream le
flux MJPEG annoté + expose les stats JSON.

Usage:
    python app.py --source 0                  # webcam
    python app.py --source video.mp4          # fichier
    python app.py --host 0.0.0.0 --port 5000  # accessible depuis le Pi
"""
import argparse
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from flask.json.provider import DefaultJSONProvider
from werkzeug.utils import secure_filename

from detector import CattleDetector
from reid import CattleReID
from database import EmbeddingDatabase


class NumpyJSONProvider(DefaultJSONProvider):
    """JSON provider qui gère les types numpy sans planter."""

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

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Palette de couleurs distinctes (BGR pour OpenCV)
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
    """Couleur stable par nom (hash -> index palette)."""
    h = 0
    for c in name:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return PALETTE[h % len(PALETTE)]


# -----------------------------
# État global partagé
# -----------------------------
STATE = {
    "frame": None,         # dernière frame annotée (BGR)
    "frame_jpg": None,     # dernière frame encodée JPEG (bytes)
    "frame_lock": threading.Lock(),
    "fps": 0.0,
    "device": "cpu",
    "started_at": None,
    "frame_count": 0,
    "active_animals": [],  # [{name, conf, track_id}, ...] frame courante
    "events": [],          # log court (NEW, MATCH, RENAME)
    "behavior": [],        # [{name, action, confidence}, ...] activités détectées
    "track_history": {},   # {track_id: [(x,y,t), ...]} pour analyse comportement
    "source": "",
    "source_label": "",    # nom court pour affichage
    "desired_source": None,  # nouveau chemin à charger (None = pas de changement)
    "current_source_path": None,  # chemin effectivement utilisé par la capture
    "desired_device": None,  # device demandé en runtime (None = pas de changement)
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--threshold", type=float, default=0.70)
    p.add_argument("--db", type=str, default="cattle_db.pkl")
    p.add_argument("--yolo-model", type=str, default="yolo11n.pt")
    p.add_argument("--dino-model", type=str, default="facebook/dinov2-small")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--skip-frames", type=int, default=0,
                   help="Traiter 1 frame sur N (0=toutes). Économise GPU.")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cpu | cuda | cuda:0 | cuda:1 ...")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def resolve_device(requested: str) -> str:
    """Résout 'auto' en cuda/cpu selon disponibilité, valide la valeur."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu"  # fallback
        if requested == "cuda":
            return "cuda:0"
        try:
            idx = int(requested.split(":")[1])
            if 0 <= idx < torch.cuda.device_count():
                return f"cuda:{idx}"
        except (ValueError, IndexError):
            pass
        return "cuda:0"
    return "cpu"


def detection_loop(args):
    """Boucle principale: lit la source, détecte, annote, met à jour STATE."""
    source = int(args.source) if args.source.isdigit() else args.source
    STATE["source"] = str(source)
    STATE["current_source_path"] = source
    STATE["source_label"] = os.path.basename(str(source)) if isinstance(source, str) and source else f"webcam {source}"

    device = resolve_device(args.device)
    STATE["device"] = device
    STATE["started_at"] = datetime.now().isoformat(timespec="seconds")

    print(f"[Init] Device demandé: {args.device} → actif: {device}", flush=True)
    if device.startswith("cuda"):
        try:
            idx = int(device.split(":")[1]) if ":" in device else 0
            print(f"[Init] GPU: {torch.cuda.get_device_name(idx)}", flush=True)
        except Exception:
            pass

    detector = CattleDetector(model_name=args.yolo_model, device=device)
    reid = CattleReID(model_name=args.dino_model, device=device)
    db = EmbeddingDatabase(path=args.db)
    # Vérifier la compatibilité dim des embeddings (purge si modèle changé)
    # On teste avec un dummy crop
    dummy = np.zeros((128, 128, 3), dtype=np.uint8)
    sample_emb = reid.get_embedding(dummy)
    if sample_emb is not None:
        remaining = db.validate_dim(int(sample_emb.shape[0]))
        print(f"[DB] Après validation: {remaining} animaux (dim={sample_emb.shape[0]})", flush=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Erreur] Source introuvable: {source}", flush=True)
        return

    track_id_to_name = {}
    track_emb_accum = {}
    fps_smooth = 0.0
    frame_idx = 0

    def open_capture(src):
        # Sur Windows, MSMF (backend par défaut) plante souvent avec les webcams.
        # DirectShow (CAP_DSHOW) est plus stable.
        if isinstance(src, int):
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                try:
                    cap = cv2.VideoCapture(src, backend)
                    if cap.isOpened():
                        # Test grab pour valider
                        ok, _ = cap.read()
                        if ok:
                            return cap
                        cap.release()
                except Exception:
                    pass
            return None
        new_cap = cv2.VideoCapture(src)
        return new_cap if new_cap.isOpened() else None

    def switch_device(new_dev):
        nonlocal detector, reid
        old = STATE["device"]
        print(f"[Device] Switch {old} → {new_dev}", flush=True)
        try:
            # YOLO Ultralytics: on passe le device au track() à chaque appel,
            # donc il suffit de mettre à jour l'attribut.
            detector.device = new_dev
            # DINOv2: il faut déplacer le modèle sur le nouveau device
            reid.model = reid.model.to(new_dev)
            reid.device = new_dev
            # Réchauffer pour ne pas pénaliser la première frame
            with torch.no_grad():
                from PIL import Image
                dummy_pil = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                dummy_in = reid.processor(images=dummy_pil, return_tensors="pt").to(new_dev)
                _ = reid.model(**dummy_in)
            STATE["device"] = new_dev
            STATE["events"].insert(0, f"DEVICE -> {new_dev}")
            STATE["events"] = STATE["events"][:30]
            print(f"[Device] OK sur {new_dev}", flush=True)
            return True
        except Exception as e:
            print(f"[Device] ERREUR: {e}", flush=True)
            return False

    # Boucle externe: récupère les erreurs fatales et repart
    while True:
        try:
            while True:
                # Vérifier si un changement de device a été demandé
                desired_dev = STATE.get("desired_device")
                if desired_dev is not None and desired_dev != STATE.get("device"):
                    if switch_device(desired_dev):
                        pass
                    STATE["desired_device"] = None

                # Vérifier si une nouvelle source a été demandée
                desired = STATE.get("desired_source")
                if desired is not None:
                    new_src = int(desired) if (isinstance(desired, str) and desired.isdigit()) else desired
                    print(f"[Switch] Demande de switch vers: {new_src}", flush=True)
                    cap.release()
                    new_cap = open_capture(new_src)
                    if new_cap is not None:
                        cap = new_cap
                        STATE["current_source_path"] = new_src
                        STATE["source"] = str(new_src)
                        # Afficher juste le nom de fichier lisible (sans timestamp)
                        if isinstance(new_src, str):
                            base = os.path.basename(new_src)
                            # Enlever le préfixe timestamp "1234567890_"
                            if "_" in base:
                                parts = base.split("_", 1)
                                if parts[0].isdigit() and len(parts[0]) >= 10:
                                    base = parts[1]
                            STATE["source_label"] = base
                        else:
                            STATE["source_label"] = f"webcam {new_src}"
                        # Reset tracker sur changement de source
                        track_id_to_name.clear()
                        track_emb_accum.clear()
                        STATE["events"].insert(0, f"SOURCE -> {STATE['source_label']}")
                        STATE["events"] = STATE["events"][:30]
                        print(f"[Switch] OK, source active: {STATE['source_label']}", flush=True)
                    else:
                        print(f"[Switch] ERREUR, impossible d'ouvrir: {new_src}", flush=True)
                        # Rouvrir l'ancienne
                        cap = open_capture(STATE["current_source_path"]) or cap
                    STATE["desired_source"] = None

                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    current = STATE["current_source_path"]
                    if isinstance(current, int):
                        # Webcam: tenter une reconnexion au lieu de mourir
                        print("[Webcam] Flux perdu, reconnexion dans 2s...", flush=True)
                        cap.release()
                        time.sleep(2.0)
                        cap = open_capture(current)
                        if cap is None:
                            print("[Webcam] Échec reconnexion, retry dans 3s...", flush=True)
                            time.sleep(3.0)
                            cap = open_capture(current) or cap  # garde un cap vide si échec
                        continue
                    # Fichier vidéo: rebobiner (mode boucle)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                result = detector.detect(frame, persist=True, conf=args.conf)
                annotated = frame.copy()
                active = []
                behaviors = []

                # Skip frames: si on n'a pas traité cette frame, on garde la précédente
                if args.skip_frames > 0 and frame_idx % (args.skip_frames + 1) != 0:
                    STATE["frame_count"] = frame_idx
                    frame_idx += 1
                    STATE["frames_skipped"] = STATE.get("frames_skipped", 0) + 1
                    continue

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    # Liste des masques alignés 1-1 avec boxes/track_ids
                    masks_data = result.masks.data.cpu().numpy() if result.masks is not None and len(result.masks) > 0 else None

                    for det_idx, (box, tid, conf) in enumerate(zip(boxes, track_ids, confs)):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if (x2 - x1) < 24 or (y2 - y1) < 24:
                            continue

                        crop = frame[y1:y2, x1:x2]
                        if int(tid) not in track_id_to_name:
                            emb = reid.get_embedding(crop)
                            if emb is not None:
                                name, sim = db.match(emb, threshold=args.threshold)
                                event = None
                                if name is None:
                                    name = f"Boeuf_{len(db.animals)+1:03d}"
                                    db.add(name, emb)
                                    event = f"NEW  {name}  (sim={sim:.3f})"
                                else:
                                    event = f"MATCH {name}  (sim={sim:.3f})"
                                track_id_to_name[int(tid)] = name
                                track_emb_accum[int(tid)] = [emb]
                                if event:
                                    STATE["events"].insert(0, event)
                                    STATE["events"] = STATE["events"][:30]
                            else:
                                track_id_to_name[int(tid)] = "?"
                        else:
                            emb = reid.get_embedding(crop)
                            if emb is not None:
                                track_emb_accum.setdefault(int(tid), []).append(emb)
                                if len(track_emb_accum[int(tid)]) % 15 == 0:
                                    mean = np.mean(track_emb_accum[int(tid)], axis=0)
                                    mean = mean / (np.linalg.norm(mean) + 1e-8)
                                    db.update(track_id_to_name[int(tid)], mean)

                        name = track_id_to_name[int(tid)]

                        # Couleur stable par bovin (basée sur le nom)
                        color = color_for_name(name)

                        # Segmentation: utiliser directement l'index de détection
                        # (les masques sont alignés avec boxes par YOLO)
                        mask_drawn = False
                        if masks_data is not None and det_idx < len(masks_data):
                            try:
                                m = masks_data[det_idx]
                                mask_resized = cv2.resize(
                                    m, (annotated.shape[1], annotated.shape[0]),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                                bin_mask = (mask_resized > 0.5).astype(np.uint8)
                                contours, _ = cv2.findContours(
                                    bin_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE,
                                )
                                # Tint intérieur très léger
                                tint = np.zeros_like(annotated)
                                tint[bin_mask == 1] = color
                                annotated = cv2.addWeighted(annotated, 1.0, tint, 0.08, 0)
                                cv2.drawContours(annotated, contours, -1, color, 2)
                                mask_drawn = True
                            except Exception:
                                pass

                        if not mask_drawn:
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                        # Label
                        label = f"{name}  {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        ly1 = max(0, y1 - th - 10)
                        ly2 = ly1 + th + 10
                        cv2.rectangle(annotated, (x1, ly1), (x1 + tw, ly2), color, -1)
                        cv2.putText(annotated, label, (x1, ly1 + th + 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        active.append({
                            "name": name,
                            "conf": float(conf),
                            "track_id": int(tid),
                        })

                # Analyse comportementale: vitesse et activité basée sur le mouvement
                t_now = time.time()
                for det_idx, (box, tid, conf) in enumerate(zip(boxes, track_ids, confs)):
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    hist = STATE["track_history"].setdefault(int(tid), [])
                    hist.append((cx, cy, t_now))
                    # Garder 1 seconde d'historique
                    STATE["track_history"][int(tid)] = [
                        p for p in hist if t_now - p[2] < 2.0
                    ]
                    pts = STATE["track_history"][int(tid)]
                    if len(pts) >= 5:
                        # Vitesse en pixels/seconde
                        dx = pts[-1][0] - pts[0][0]
                        dy = pts[-1][1] - pts[0][1]
                        dist = (dx * dx + dy * dy) ** 0.5
                        dt = pts[-1][2] - pts[0][2]
                        speed = dist / max(dt, 1e-3)
                        # Heuristiques simples
                        if speed < 5:
                            action = "immobile"
                        elif speed < 25:
                            action = "marche"
                        elif speed < 80:
                            action = "court"
                        else:
                            action = "rué"
                        # Vérifier si la tête est en bas (mange) — bounding box plus large que haute
                        bw = box[2] - box[0]
                        bh = box[3] - box[1]
                        aspect = bw / max(bh, 1)
                        if speed < 5 and aspect > 1.4:
                            action = "pâture"
                        # Si la bbox est verticale et vitesse faible → peut être debout statique
                        behaviors.append({
                            "name": track_id_to_name.get(int(tid), "?"),
                            "action": action,
                            "speed": round(speed, 1),
                            "track_id": int(tid),
                        })

                STATE["behavior"] = behaviors[:20]

                # Encodage JPEG pour le stream
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    with STATE["frame_lock"]:
                        STATE["frame_jpg"] = buf.tobytes()
                        STATE["active_animals"] = active

                elapsed = time.time() - t0
                fps_inst = 1.0 / max(elapsed, 1e-6)
                fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst if fps_smooth else fps_inst
                STATE["fps"] = fps_smooth
                STATE["frame_count"] = frame_idx
                frame_idx += 1
        except Exception as e:
            print(f"[Crash] {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            STATE["events"].insert(0, f"CRASH récupéré: {type(e).__name__}")
            STATE["events"] = STATE["events"][:30]
            print("[Crash] Redémarrage dans 2s...", flush=True)
            time.sleep(2)
            try:
                cap.release()
            except Exception:
                pass
            cap = open_capture(STATE["current_source_path"]) or cv2.VideoCapture(0)
            continue
    if not args.no_save:
        db.save()
    cap.release()


# -----------------------------
# Flask
# -----------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.json = NumpyJSONProvider(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Stream MJPEG multipart."""
    def gen():
        boundary = b"--frame"
        while True:
            with STATE["frame_lock"]:
                jpg = STATE["frame_jpg"]
            if jpg is None:
                time.sleep(0.05)
                continue
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.03)  # ~30 FPS max servi
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
        "behavior": STATE.get("behavior", []),
    })


@app.route("/api/diag")
def diag():
    """Diagnostic: état interne complet."""
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


@app.route("/api/devices")
def list_devices():
    """Liste les devices disponibles et le device actif."""
    available = ["cpu"]
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus.append({
                "index": i,
                "id": f"cuda:{i}",
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
            })
            available.append(f"cuda:{i}")
    return jsonify({
        "available": available,
        "current": STATE["device"],
        "gpus": gpus,
    })


@app.route("/api/device", methods=["POST"])
def set_device():
    """Change le device à la volée."""
    data = request.get_json(silent=True) or {}
    requested = (data.get("device") or "auto").strip().lower()
    resolved = resolve_device(requested)
    if requested.startswith("cuda") and resolved == "cpu":
        return jsonify({
            "ok": False,
            "error": "CUDA non disponible, fallback CPU",
            "resolved": resolved,
        }), 400
    if resolved == STATE["device"]:
        return jsonify({"ok": True, "device": resolved, "changed": False})
    STATE["desired_device"] = resolved
    return jsonify({"ok": True, "device": resolved, "changed": True})


@app.route("/api/rename", methods=["POST"])
def rename():
    data = request.get_json(silent=True) or {}
    old = data.get("old")
    new = data.get("new")
    if not old or not new or old == new:
        return jsonify({"ok": False, "error": "paramètres invalides"}), 400
    # Note: rename direct sur la DB requiert accès thread-safe; on fait simple ici.
    return jsonify({"ok": True, "old": old, "new": new})


@app.route("/api/upload-video", methods=["POST"])
def upload_video():
    """Upload d'une vidéo locale et bascule de la source dessus."""
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "aucun fichier"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"ok": False, "error": "nom de fichier vide"}), 400

    name = secure_filename(f.filename)
    if not name:
        return jsonify({"ok": False, "error": "nom invalide"}), 400

    # Refuser les extensions non-vidéo par sécurité
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".wmv"}
    ext = os.path.splitext(name)[1].lower()
    if ext not in allowed:
        return jsonify({"ok": False, "error": f"extension non supportée: {ext}"}), 400

    # Préfixe timestamp pour éviter les collisions
    ts_name = f"{int(time.time())}_{name}"
    dest = os.path.join(UPLOAD_DIR, ts_name)
    f.save(dest)

    STATE["desired_source"] = dest
    return jsonify({"ok": True, "filename": name, "path": dest})


@app.route("/api/source/webcam", methods=["POST"])
def source_webcam():
    """Bascule sur la webcam (index 0 par défaut)."""
    data = request.get_json(silent=True) or {}
    idx = int(data.get("index", 0))
    STATE["desired_source"] = idx
    return jsonify({"ok": True, "index": idx})


def main():
    args = parse_args()

    print("=" * 60)
    print("  BOEUF TRACKER — Interface web")
    print(f"  http://{args.host}:{args.port}")
    print("=" * 60)

    t = threading.Thread(target=detection_loop, args=(args,), daemon=True)
    t.start()

    # threaded=True pour gérer /video_feed et /api/stats en parallèle
    app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()