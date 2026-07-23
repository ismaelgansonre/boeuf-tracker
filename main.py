"""
main.py
-------
Système de reconnaissance de bovins: détection YOLOv11 + tracking intra-vidéo
+ Re-ID cross-vidéo via DINOv2.

Usage:
    python main.py                                  # menu interactif
    python main.py --source 0                       # webcam
    python main.py --source video.mp4               # fichier vidéo
    python main.py --source rtsp://...              # flux réseau (futur: Pi)
"""
import argparse
import time
import cv2
import numpy as np
import torch

# Use new OOP structure
from core.detector import CattleDetector
from core.reid import CattleReID
from core.database import EmbeddingDatabase
from core.capture import CaptureManager
from models.breed import classify_breed
from utils.names import get_name_generator, next_bovin_key
from utils.console import banner, info, ok, err
from utils.state import color_for_name


def parse_args():
    p = argparse.ArgumentParser(
        description="Reconnaissance de bovins: détection + tracking + Re-ID"
    )
    p.add_argument("--source", type=str, default=None,
                   help="Chemin vidéo, index webcam (0), ou URL RTSP")
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Seuil cosine pour Re-ID (0-1, plus haut = plus strict)")
    p.add_argument("--db", type=str, default="cattle_db.pkl",
                   help="Fichier base de données d'embeddings")
    p.add_argument("--yolo-model", type=str, default="yolo11n.pt",
                   help="Modèle YOLO (yolo11n/s/m/l/x.pt ou votre .pt fine-tuné)")
    p.add_argument("--dino-model", type=str, default="facebook/dinov2-small",
                   help="Modèle DINOv2 pour embeddings")
    p.add_argument("--conf", type=float, default=0.4,
                   help="Seuil de confiance YOLO")
    p.add_argument("--no-save", action="store_true",
                   help="Ne pas sauvegarder la base à la fin")
    return p.parse_args()


def choose_source_interactive() -> str:
    print("\n[1] Charger une vidéo")
    print("[2] Utiliser la webcam")
    choice = input("Choix [1/2]: ").strip()
    if choice == "1":
        path = input("Chemin de la vidéo: ").strip().strip('"').strip("'")
        return path
    return "0"


def main():
    args = parse_args()

    print("=" * 64)
    print("  SYSTÈME DE RECONNAISSANCE DE BOVINS")
    print("  YOLOv11 (détection + tracking) + DINOv2 (Re-ID)")
    print("=" * 64)

    # Source
    source = args.source if args.source is not None else choose_source_interactive()
    is_camera = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
    if is_camera:
        source = int(source)
        print(f"\n[Source] Webcam (index {source})")
    else:
        print(f"\n[Source] {source}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    if device == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # Modèles
    detector = CattleDetector(model_name=args.yolo_model, device=device)
    reid = CattleReID(model_name=args.dino_model, device=device)
    db = EmbeddingDatabase(path=args.db)

    # Mémoire de session
    track_id_to_name: dict[int, str] = {}
    track_emb_accum: dict[int, list] = {}
    last_known_tid: int | None = None

    # Capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Erreur] Impossible d'ouvrir la source: {source}")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"\n[Info] Démarrage de la boucle vidéo.")
    print("       Touches: [q] quitter | [s] screenshot | [n] renommer dernier bovin")

    frame_count = 0
    fps_smooth = 0.0
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                if not is_camera:
                    print("[Info] Fin de la vidéo.")
                else:
                    print("[Erreur] Flux caméra perdu.")
                break

            # Détection + tracking
            result = detector.detect(frame, persist=True, conf=args.conf)
            annotated = frame.copy()

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                active_ids = set()
                for box, tid, conf in zip(boxes, track_ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                    if (x2 - x1) < 24 or (y2 - y1) < 24:
                        continue

                    active_ids.add(int(tid))
                    crop = frame[y1:y2, x1:x2]

                    # Nouveau track_id -> identifier
                    if tid not in track_id_to_name:
                        emb = reid.get_embedding(crop)
                        if emb is not None:
                            name, sim = db.match(emb, threshold=args.threshold)
                            if name is None:
                                new_idx = len(db.animals) + 1
                                name = f"Boeuf_{new_idx:03d}"
                                db.add(name, emb)
                                print(f"  [NEW]   {name}  (sim_max={sim:.3f})")
                            else:
                                print(f"  [MATCH] {name}  (sim={sim:.3f})")
                            track_id_to_name[int(tid)] = name
                            track_emb_accum[int(tid)] = [emb]
                            last_known_tid = int(tid)
                        else:
                            track_id_to_name[int(tid)] = "?"
                    else:
                        # Track connu -> accumuler pour stabiliser l'embedding
                        emb = reid.get_embedding(crop)
                        if emb is not None:
                            track_emb_accum.setdefault(int(tid), []).append(emb)
                            if len(track_emb_accum[int(tid)]) % 15 == 0:
                                mean = np.mean(track_emb_accum[int(tid)], axis=0)
                                mean = mean / (np.linalg.norm(mean) + 1e-8)
                                db.update(track_id_to_name[int(tid)], mean)

                    name = track_id_to_name[int(tid)]
                    color = (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{name}  {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                    cv2.putText(annotated, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Nettoyage des tracks disparus (garder mémoire courte)
                # On garde tout pour l'instant — l'embedding se stabilise en DB

            # HUD
            elapsed = time.time() - t0
            fps_inst = 1.0 / max(elapsed, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst if fps_smooth else fps_inst
            hud = f"FPS: {fps_smooth:.1f}  |  Device: {device}  |  DB: {len(db.animals)} animaux"
            cv2.putText(annotated, hud, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Cattle Re-ID  (q=quit, s=screenshot, n=rename)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                path = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(path, annotated)
                print(f"  [Screenshot] {path}")
            elif key == ord("n"):
                if last_known_tid is not None and last_known_tid in track_id_to_name:
                    old = track_id_to_name[last_known_tid]
                    new = input(f"  Nouveau nom pour {old}: ").strip()
                    if new and new != old:
                        if db.rename(old, new):
                            track_id_to_name[last_known_tid] = new
                            print(f"  [Rename] {old} -> {new}")
                        else:
                            print(f"  [Rename] Échec (nom déjà pris?)")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[Stop] Ctrl+C")
    finally:
        if not args.no_save:
            db.save()
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[Final] {len(db.animals)} animaux en base.")


if __name__ == "__main__":
    main()