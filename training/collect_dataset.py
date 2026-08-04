"""
training/collect_dataset.py
---------------------------
Étape 1 du pipeline d'apprentissage de la posture.

Rejoue une ou plusieurs vidéos avec le VRAI pipeline (détecteur + posture) et,
pour chaque bovin suivi, exporte :

  - une imagette JPEG rangée dans un dossier correspondant à la posture
    PROPOSÉE par les règles actuelles (couché / debout / pature) ;
  - l'embedding MegaDescriptor de cette imagette (le même vecteur que celui
    calculé pour la ré-identification — coût nul à l'inférence plus tard).

Tu n'as ensuite qu'à CORRIGER les erreurs dans le Finder : glisse les vignettes
mal classées d'un dossier à l'autre. Le dossier dans lequel se trouve une
imagette au moment de l'entraînement fait foi (cf. train_posture.py).

POURQUOI un pré-étiquetage plutôt qu'un étiquetage à zéro
--------------------------------------------------------
Les règles ont raison la plupart du temps. Corriger une minorité d'erreurs est
bien plus rapide qu'étiqueter des centaines d'imagettes de zéro, et le dataset
reste dans TON domaine (bœufs bruns, smartphone) — c'est ce que les datasets
publics ne donnent pas.

Usage
-----
    python training/collect_dataset.py \
        --videos samples/IMG_3544.mov samples/IMG_3550.mov samples/IMG_3553.mov \
        --out training/dataset --every 8 --max-per-track 40

Options
-------
  --every N         : n'échantillonne qu'une frame sur N par piste (évite les
                      quasi-doublons). Défaut 8.
  --max-per-track M : plafond d'imagettes par piste et par vidéo. Défaut 40.
  --imgsz / --conf  : réglages du détecteur (mêmes que l'appli).
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np

# Rendre le package racine importable quel que soit le cwd d'appel.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from detector import CattleDetectorMLX, CattleDetector  # noqa: E402
from reid import CattleReID  # noqa: E402
from posture import MaskHeadAnalyzer, HeadMotionTracker, overlap_fractions  # noqa: E402
from processor import _mlx_available, resolve_device  # noqa: E402

# Postures proposées → dossiers de tri. "autre" recueille ce que tu veux
# écarter (silhouette tronquée, imagette illisible) : ces imagettes ne serviront
# pas à l'entraînement.
LABELS = ("couché", "debout", "pature", "autre")


def _proposed_label(lying: bool, head_down: bool) -> str:
    if lying:
        return "couché"
    if head_down:
        return "pature"
    return "debout"


def _build_detector(imgsz: int):
    """Même logique de sélection que l'appli : MLX si dispo, sinon PyTorch."""
    if _mlx_available():
        return CattleDetectorMLX(), "mps"
    dev = resolve_device("auto")
    return CattleDetector(model_name="yolo11s-seg.pt", device=dev), dev


def collect(videos, out_dir, every, max_per_track, imgsz, conf):
    os.makedirs(out_dir, exist_ok=True)
    for lab in LABELS:
        os.makedirs(os.path.join(out_dir, lab), exist_ok=True)

    detector, reid_dev = _build_detector(imgsz)
    reid = CattleReID(device=reid_dev)

    # Embeddings persistés à part : {basename_imagette: vecteur float32}.
    # On repartira d'un fichier existant pour pouvoir enchaîner plusieurs runs.
    emb_path = os.path.join(out_dir, "embeddings.pkl")
    embeddings: dict[str, np.ndarray] = {}
    if os.path.exists(emb_path):
        with open(emb_path, "rb") as f:
            embeddings = pickle.load(f)

    total_saved = 0
    for video in videos:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"[skip] vidéo illisible: {video}")
            continue
        tag = os.path.splitext(os.path.basename(video))[0]
        head = MaskHeadAnalyzer()
        motion = HeadMotionTracker()
        seen_count: dict[int, int] = {}   # frames vues par piste
        saved_count: dict[int, int] = {}  # imagettes sauvées par piste
        fidx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            r = detector.detect(frame, conf=conf, imgsz=imgsz)
            if r.boxes is None or r.boxes.id is None or len(r.boxes) == 0:
                fidx += 1
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            tids = r.boxes.id.int().cpu().numpy()
            masks = r.masks.data.cpu().numpy() if r.masks is not None else None
            occl = overlap_fractions(boxes)
            H, W = frame.shape[:2]

            for i, (box, tid) in enumerate(zip(boxes, tids)):
                tid = int(tid)
                seen_count[tid] = seen_count.get(tid, 0) + 1
                # Posture proposée (mêmes calculs que l'appli).
                lying = head_down = False
                truncated = False
                if masks is not None and i < len(masks):
                    mr = cv2.resize(masks[i], (W, H), interpolation=cv2.INTER_NEAREST)
                    x1, y1, x2, y2 = box.astype(int)
                    roi = (mr > 0.5)[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
                    truncated = (y2 >= H - 4) or (i < len(occl) and occl[i] > 0.25)
                    st = head.analyze(roi, truncated_bottom=bool(truncated))
                    head_down = motion.update(tid, st)
                    lying = motion.is_lying(tid)

                # Échantillonnage : 1 frame sur `every`, plafonné par piste.
                if seen_count[tid] % every != 0:
                    continue
                if saved_count.get(tid, 0) >= max_per_track:
                    continue

                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if (x2 - x1) < 24 or (y2 - y1) < 24:
                    continue
                crop = frame[y1:y2, x1:x2]
                emb = reid.get_embedding(crop)
                if emb is None:
                    continue

                label = "autre" if truncated else _proposed_label(lying, head_down)
                fname = f"{tag}_f{fidx}_t{tid}.jpg"
                cv2.imwrite(os.path.join(out_dir, label, fname), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                embeddings[fname] = np.asarray(emb, dtype=np.float32)
                saved_count[tid] = saved_count.get(tid, 0) + 1
                total_saved += 1
            fidx += 1
        cap.release()
        print(f"[ok] {tag}: {sum(saved_count.values())} imagettes")

    with open(emb_path, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"\n{total_saved} imagettes exportées dans {out_dir}/")
    print("Postures proposées (à corriger dans le Finder) :")
    for lab in LABELS:
        n = len(os.listdir(os.path.join(out_dir, lab)))
        print(f"  {lab:8s}: {n}")
    print("\nÉtape suivante : corrige les erreurs en glissant les imagettes "
          "entre dossiers, puis lance training/train_posture.py")


def main():
    ap = argparse.ArgumentParser(description="Collecte le dataset de posture.")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", default="training/dataset")
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--max-per-track", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    collect(args.videos, args.out, args.every, args.max_per_track,
            args.imgsz, args.conf)


if __name__ == "__main__":
    main()
