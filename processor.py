"""
processor.py
------------
Boucle principale de détection : lit la source, détecte + track + identifie
les bovins, annote la frame, met à jour STATE.

Auto-recovery en cas de crash (la boucle redémarre automatiquement).
"""
import os
import time
import threading
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image

from detector import CattleDetector
from reid import CattleReID
from database import EmbeddingDatabase
from state import STATE, color_for_name, reset_for_new_source
from capture import open_capture, source_label


def annotate_frame(annotated, masks_data, det_idx, x1, y1, x2, y2, color):
    """Dessine le masque de segmentation (silhouette) si dispo, sinon rectangle."""
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
                bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            # Tint intérieur subtil (8%) + contour net
            tint = np.zeros_like(annotated)
            tint[bin_mask == 1] = color
            annotated[:] = cv2.addWeighted(annotated, 1.0, tint, 0.08, 0)
            cv2.drawContours(annotated, contours, -1, color, 2)
            mask_drawn = True
        except Exception:
            pass
    if not mask_drawn:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    return annotated


def analyze_behavior(boxes, track_ids, t_now, frame_shape=None):
    """Catégorise l'activité: pâture, boit, couché, immobile, marche, court, rué."""
    behaviors = []
    frame_h = frame_shape[0] if frame_shape is not None else 1080
    for box, tid in zip(boxes, track_ids):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        aspect = bw / max(bh, 1)
        rel_y = cy / frame_h  # 0 = haut, 1 = bas

        hist = STATE["track_history"].setdefault(int(tid), [])
        hist.append((cx, cy, t_now))
        STATE["track_history"][int(tid)] = [p for p in hist if t_now - p[2] < 5.0]
        pts = STATE["track_history"][int(tid)]
        if len(pts) < 3:
            continue

        # Vitesse instantanée
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        dist = (dx * dx + dy * dy) ** 0.5
        dt = max(pts[-1][2] - pts[0][2], 1e-3)
        speed = dist / dt
        # Temps sans bouger significativement
        total_disp = sum(
            ((pts[i][0] - pts[i-1][0]) ** 2 + (pts[i][1] - pts[i-1][1]) ** 2) ** 0.5
            for i in range(1, len(pts))
        )
        immobile_since = pts[0][2] if total_disp < 30 else None

        action = None
        # 1. Couché: bbox très large, immobile depuis longtemps
        if aspect > 1.7 and speed < 5 and immobile_since and (t_now - immobile_since) > 3.0:
            action = "couché"
        # 2. Pâture: tête en bas (bbox large), déplacement lent
        elif speed < 6 and aspect > 1.4:
            action = "pâture"
        # 3. Boit: tête en bas, immobile, en bas de la frame (présence point d'eau)
        elif speed < 4 and aspect > 1.3 and rel_y > 0.6:
            action = "boit"
        # 4. Vitesse
        elif speed < 5:
            action = "immobile"
        elif speed < 25:
            action = "marche"
        elif speed < 80:
            action = "court"
        else:
            action = "rué"

        behaviors.append({
            "name": STATE.get("_track_names", {}).get(int(tid), "?"),
            "action": action,
            "speed": round(float(speed), 1),
            "track_id": int(tid),
            "aspect": round(float(aspect), 2),
        })
    return behaviors


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu"
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
    """Boucle principale avec auto-recovery."""
    source = int(args.source) if args.source.isdigit() else args.source
    STATE["source"] = str(source)
    STATE["current_source_path"] = source
    STATE["source_label"] = source_label(source)
    STATE["started_at"] = datetime.now().isoformat(timespec="seconds")

    device = resolve_device(args.device)
    STATE["device"] = device
    print(f"[Init] Device: {device}", flush=True)
    if device.startswith("cuda"):
        try:
            idx = int(device.split(":")[1]) if ":" in device else 0
            print(f"[Init] GPU: {torch.cuda.get_device_name(idx)}", flush=True)
        except Exception:
            pass

    # Modèles
    detector = CattleDetector(model_name=args.yolo_model, device=device)
    reid = CattleReID(model_name=args.dino_model, device=device)
    db = EmbeddingDatabase(path=args.db, reid_engine=reid)

    # Validation compatibilité dim
    dummy_crop = np.zeros((128, 128, 3), dtype=np.uint8)
    sample_emb = reid.get_embedding(dummy_crop)
    expected_dim = int(sample_emb.shape[0]) if sample_emb is not None else reid.TOTAL_DIM
    remaining = db.validate_dim(expected_dim)
    print(f"[DB] {remaining} animaux (dim={expected_dim})", flush=True)

    # Sync initial settings to STATE (sinon les boutons UI ne savent pas l'état réel)
    STATE["yolo_model_current"] = args.yolo_model
    STATE["imgsz_current"] = args.imgsz
    STATE["embed_every_current"] = args.embed_every
    STATE["threshold_current"] = args.threshold
    STATE["conf_current"] = args.conf

    # Capture initiale
    cap = open_capture(source)
    if cap is None:
        print(f"[Erreur] Source introuvable: {source}", flush=True)
        return

    track_id_to_name = {}
    STATE["_track_names"] = track_id_to_name  # pour behavior
    track_emb_accum = {}
    fps_smooth = 0.0
    frame_idx = 0

    def switch_device(new_dev):
        try:
            print(f"[Device] Switch {STATE['device']} → {new_dev}", flush=True)
            detector.device = new_dev
            reid.model = reid.model.to(new_dev)
            reid.device = new_dev
            with torch.no_grad():
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

    def apply_desired_settings():
        """Applique les changements demandés via /api/settings à chaud."""
        nonlocal detector

        # imgsz (changement instantané, pas de reload)
        d = STATE.get("desired_imgsz")
        if d is not None and d != STATE.get("imgsz_current"):
            args.imgsz = d
            STATE["imgsz_current"] = d
            STATE["desired_imgsz"] = None
            STATE["events"].insert(0, f"IMGSZ -> {d}")
            STATE["events"] = STATE["events"][:30]
            print(f"[Settings] imgsz -> {d}", flush=True)

        # embed_every (pas de reload, juste fréquence)
        d = STATE.get("desired_embed_every")
        if d is not None and d != STATE.get("embed_every_current"):
            STATE["embed_every_current"] = d
            STATE["desired_embed_every"] = None
            STATE["events"].insert(0, f"EMBED_EVERY -> {d}")
            STATE["events"] = STATE["events"][:30]
            print(f"[Settings] embed_every -> {d}", flush=True)

        # threshold (pas de reload)
        d = STATE.get("desired_threshold")
        if d is not None and abs(d - STATE.get("threshold_current", 0)) > 1e-6:
            args.threshold = d
            STATE["threshold_current"] = d
            STATE["desired_threshold"] = None
            STATE["events"].insert(0, f"THRESHOLD -> {d:.2f}")
            STATE["events"] = STATE["events"][:30]
            print(f"[Settings] threshold -> {d}", flush=True)

        # conf (pas de reload)
        d = STATE.get("desired_conf")
        if d is not None and abs(d - STATE.get("conf_current", 0)) > 1e-6:
            args.conf = d
            STATE["conf_current"] = d
            STATE["desired_conf"] = None
            STATE["events"].insert(0, f"CONF -> {d:.2f}")
            STATE["events"] = STATE["events"][:30]
            print(f"[Settings] conf -> {d}", flush=True)

        # yolo_model (reload du modèle, ~2-5s de freeze)
        d = STATE.get("desired_yolo_model")
        if d is not None and d != STATE.get("yolo_model_current"):
            print(f"[YOLO] Reload {STATE.get('yolo_model_current')} -> {d}", flush=True)
            STATE["events"].insert(0, f"YOLO reload -> {d}")
            STATE["events"] = STATE["events"][:30]
            try:
                old = detector
                new_det = CattleDetector(model_name=d, device=old.device, half=old.half)
                detector = new_det
                STATE["yolo_model_current"] = d
                STATE["desired_yolo_model"] = None
                STATE["events"].insert(0, f"YOLO OK: {d}")
                STATE["events"] = STATE["events"][:30]
                print(f"[YOLO] Reload OK", flush=True)
            except Exception as e:
                STATE["desired_yolo_model"] = None  # clear to avoid loop
                STATE["events"].insert(0, f"YOLO ERREUR: {e}")
                STATE["events"] = STATE["events"][:30]
                print(f"[YOLO] Reload ERREUR: {e}", flush=True)

    # Boucle externe: récupère les crashes
    while True:
        try:
            while True:
                # Applique les changements live (imgsz, embed_every, threshold, conf, model)
                apply_desired_settings()

                # Device switch
                desired_dev = STATE.get("desired_device")
                if desired_dev is not None and desired_dev != STATE.get("device"):
                    switch_device(desired_dev)
                    STATE["desired_device"] = None

                # Source switch
                desired = STATE.get("desired_source")
                if desired is not None:
                    new_src = int(desired) if (isinstance(desired, str) and desired.isdigit()) else desired
                    print(f"[Switch] -> {new_src}", flush=True)
                    cap.release()
                    new_cap = open_capture(new_src)
                    if new_cap is not None:
                        cap = new_cap
                        STATE["current_source_path"] = new_src
                        STATE["source"] = str(new_src)
                        STATE["source_label"] = source_label(new_src)
                        track_id_to_name.clear()
                        track_emb_accum.clear()
                        STATE["_track_names"] = track_id_to_name
                        reset_for_new_source()
                        print(f"[Switch] OK: {STATE['source_label']}", flush=True)
                    else:
                        cap = open_capture(STATE["current_source_path"]) or cap
                        print(f"[Switch] ERREUR ouverture", flush=True)
                    STATE["desired_source"] = None

                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    src = STATE["current_source_path"]
                    if isinstance(src, int):
                        # Webcam: reconnect
                        try:
                            cap.release()
                        except Exception:
                            pass
                        time.sleep(2.0)
                        new_cap = open_capture(src)
                        if new_cap is None:
                            time.sleep(3.0)
                            new_cap = open_capture(src)
                        cap = new_cap or cap
                    else:
                        # Fichier: rebobiner
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue  # on re-tentera à la prochaine itération

                # Skip frames pour économie GPU
                if args.skip_frames > 0 and frame_idx % (args.skip_frames + 1) != 0:
                    STATE["frame_count"] = frame_idx
                    frame_idx += 1
                    continue

                # Détection + tracking + segmentation
                result = detector.detect(frame, persist=True, conf=args.conf, imgsz=args.imgsz)
                annotated = frame.copy()
                active = []

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    masks_data = (
                        result.masks.data.cpu().numpy()
                        if result.masks is not None and len(result.masks) > 0
                        else None
                    )

                    # Noms déjà attribués dans cette frame (pour exclusion)
                    frame_names = {
                        track_id_to_name[t]
                        for t in track_ids
                        if int(t) in track_id_to_name and track_id_to_name[int(t)] != "?"
                    }

                    for det_idx, (box, tid, conf) in enumerate(zip(boxes, track_ids, confs)):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if (x2 - x1) < 24 or (y2 - y1) < 24:
                            continue

                        crop = frame[y1:y2, x1:x2]
                        if int(tid) not in track_id_to_name:
                            emb = reid.get_embedding(crop)
                            if emb is not None and emb.shape[0] == expected_dim:
                                # Exclure les noms déjà utilisés dans cette frame
                                name, sim = db.match(
                                    emb,
                                    threshold=args.threshold,
                                    exclude=frame_names,
                                )
                                event = None
                                if name is None:
                                    name = f"Boeuf_{len(db.animals) + 1:03d}"
                                    db.add(name, emb)
                                    event = f"NEW  {name}  (sim_max={sim:.3f})"
                                else:
                                    event = f"MATCH {name}  (sim={sim:.3f})"
                                track_id_to_name[int(tid)] = name
                                track_emb_accum[int(tid)] = [emb]
                                STATE["_track_names"] = track_id_to_name
                                # Ajouter ce nom aux noms déjà pris pour les suivants
                                frame_names.add(name)
                                if event:
                                    STATE["events"].insert(0, event)
                                    STATE["events"] = STATE["events"][:30]
                            else:
                                track_id_to_name[int(tid)] = "?"
                                STATE["_track_names"] = track_id_to_name
                        else:
                            # Re-embedding seulement tous les N frames (ByteTrack gère déjà
                            # le tracking motion, DINOv2 ne sert qu'à rafraïchir l'identité).
                            reid_every = max(1, int(STATE.get("embed_every_current", 30)))
                            if frame_idx % reid_every == 0:
                                emb = reid.get_embedding(crop)
                                if emb is not None and emb.shape[0] == expected_dim:
                                    buf = track_emb_accum.setdefault(int(tid), [])
                                    buf.append(emb)
                                    # Garde seulement les 4 derniers pour éviter l'explosion mémoire
                                    if len(buf) > 4:
                                        track_emb_accum[int(tid)] = buf[-4:]
                                        buf = track_emb_accum[int(tid)]
                                    # Moyenne des 2 derniers (plus réactif à la pose actuelle)
                                    if len(buf) >= 2:
                                        stacked = np.stack(buf[-2:]).astype(np.float64)
                                        mean = stacked.mean(axis=0)
                                        mean = mean / (np.linalg.norm(mean) + 1e-8)
                                        db.update(track_id_to_name[int(tid)], mean.astype(np.float32))

                        name = track_id_to_name[int(tid)]
                        color = color_for_name(name)
                        annotated = annotate_frame(
                            annotated, masks_data, det_idx,
                            x1, y1, x2, y2, color,
                        )

                        # Label
                        label = f"{name}  {float(conf):.2f}"
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

                    # Comportement (sur la dernière frame traitée)
                    STATE["behavior"] = analyze_behavior(boxes, track_ids, time.time(), frame.shape)

                # Encodage JPEG
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
            STATE["events"].insert(0, f"CRASH: {type(e).__name__}")
            STATE["events"] = STATE["events"][:30]
            time.sleep(2)
            try:
                cap.release()
            except Exception:
                pass
            cap = open_capture(STATE["current_source_path"]) or cv2.VideoCapture(0)
            continue


def _recover_capture(cap, src, ret):
    """Helper legacy (gardé pour compat)."""
    if ret:
        return cap
    if isinstance(src, int):
        cap.release()
        time.sleep(2.0)
        new_cap = open_capture(src)
        if new_cap is None:
            time.sleep(3.0)
            new_cap = open_capture(src)
        return new_cap or cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap


def start_detection_thread(args) -> threading.Thread:
    """Démarre le thread de détection. À appeler une seule fois au boot."""
    t = threading.Thread(target=detection_loop, args=(args,), daemon=True)
    t.start()
    return t


# Imports internes pour switch_device