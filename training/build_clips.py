"""
training/build_clips.py
-----------------------
Étape 1 de la méthode X3D (vidéo) : reconstruire des CLIPS DENSES à partir des
vidéos sources, en réutilisant les labels déjà corrigés dans training/dataset/.

POURQUOI
--------
Le dataset d'images (training/dataset/{pature,debout,couché}) a été échantillonné
avec --every 8 : les crops sont ESPACÉS. X3D a besoin de frames CONSÉCUTIVES
pour lire le mouvement. On rejoue donc les vidéos SANS sous-échantillonnage et,
autour de chaque frame étiquetée, on extrait une fenêtre de T frames continues du
même bovin (même track id), chacune recadrée sur sa bbox → un tenseur clip.

Le label du clip = le dossier où se trouve l'imagette de sa frame centrale
(couché / debout / pature). "autre" est ignoré.

SORTIE
------
    training/clips/<label>/<video>_t<track>_f<frame>.npy   # (T, H, W, 3) uint8
On stocke en .npy (rapide à charger pour l'entraînement) + un index CSV.

Usage
-----
    python training/build_clips.py \
        --videos samples/IMG_3544.mov samples/IMG_3545.mov ... \
        --dataset training/dataset --out training/clips \
        --clip-len 16 --size 182

Note : clip-len=16, size=182 = config d'entrée X3D-M (voir train_x3d.py).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from detector import CattleDetector, CattleDetectorMLX  # noqa: E402
from processor import _mlx_available, resolve_device  # noqa: E402

LABELS = ("couché", "debout", "pature")  # "autre" exclu
_FN = re.compile(r"^(?P<vid>.+)_f(?P<frame>\d+)_t(?P<track>\d+)\.jpg$")


def index_labels(dataset_dir: str) -> dict[tuple[str, int], dict[int, str]]:
    """(video, track) -> {frame: label} d'après le rangement des imagettes."""
    out: dict[tuple[str, int], dict[int, str]] = defaultdict(dict)
    for label in LABELS:
        d = os.path.join(dataset_dir, label)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            m = _FN.match(name)
            if not m:
                continue
            vid = m.group("vid")
            frame = int(m.group("frame"))
            track = int(m.group("track"))
            out[(vid, track)][frame] = label
    return out


def make_detector():
    device = resolve_device()
    if _mlx_available():
        return CattleDetectorMLX(), device
    return CattleDetector(device=device), device


def build_for_video(video_path: str, labels_idx, out_dir: str,
                    clip_len: int, size: int, writer) -> int:
    """Rejoue la vidéo en tracking dense et exporte les clips étiquetés."""
    vid = os.path.splitext(os.path.basename(video_path))[0]
    # frames étiquetées pour cette vidéo, par track
    wanted = {(t, f): lab
              for (v, t), fl in labels_idx.items() if v == vid
              for f, lab in fl.items()}
    if not wanted:
        return 0

    detector, _ = make_detector()
    cap = cv2.VideoCapture(video_path)

    # buffer glissant par track : track -> deque des (frame_idx, crop)
    from collections import deque
    half = clip_len // 2
    buffers: dict[int, deque] = defaultdict(lambda: deque(maxlen=clip_len))
    written = 0
    frame_idx = -1

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1

        # tracking dense (persist=True conserve les IDs entre frames).
        # detect() renvoie un objet Ultralytics Results.
        res = detector.detect(frame_bgr, persist=True)
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            continue
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            crop = frame_bgr[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (size, size))
            buffers[tid].append((frame_idx, crop))

            # une frame étiquetée devient le CENTRE d'un clip dès que le buffer
            # contient la fenêtre complète autour d'elle
            center = frame_idx - half
            key = (tid, center)
            if key in wanted and len(buffers[tid]) == clip_len:
                clip = np.stack([c for _, c in buffers[tid]], axis=0)  # (T,H,W,3)
                label = wanted[key]
                dst_dir = os.path.join(out_dir, label)
                os.makedirs(dst_dir, exist_ok=True)
                fn = f"{vid}_t{tid}_f{center}.npy"
                np.save(os.path.join(dst_dir, fn), clip)
                writer.writerow([os.path.join(label, fn), label, vid, tid, center])
                written += 1

    cap.release()
    print(f"[clips] {vid}: {written} clips écrits")
    return written


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--dataset", default="training/dataset")
    p.add_argument("--out", default="training/clips")
    p.add_argument("--clip-len", type=int, default=16)
    p.add_argument("--size", type=int, default=182)
    args = p.parse_args()

    labels_idx = index_labels(args.dataset)
    os.makedirs(args.out, exist_ok=True)
    total = 0
    with open(os.path.join(args.out, "index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "video", "track", "frame"])
        for v in args.videos:
            total += build_for_video(v, labels_idx, args.out, args.clip_len, args.size, w)
    print(f"[clips] TOTAL : {total} clips → {args.out}")


if __name__ == "__main__":
    main()
