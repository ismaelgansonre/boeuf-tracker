"""
training/train_behavior_video.py
--------------------------------
Entraîne un modèle de COMPORTEMENT VIDÉO sur le dataset CVB, EN LOCAL (Mac/MPS).
Aucune dépendance Colab / pytorchvideo : on utilise torchvision (maintenu).

Modèle : R(2+1)D-18 pré-entraîné Kinetics-400 (3D CNN vidéo léger, 112×112).
On n'affine que la tête sur les comportements CVB.

Usage :
    .venv/bin/python training/train_behavior_video.py \
        --cvb external/cvb_full/58916v001/data --epochs 15

Sortie : behavior_video.pt  (poids + labels + config)
         → chargé par behavior_video.py dans le pipeline.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

# behN (nom de dossier CVB) → nom lisible (cf. behaviour_list.pbtx)
BEH_MAP = {
    1: "none", 2: "grazing", 3: "walking", 4: "ruminating-standing",
    5: "ruminating-lying", 6: "resting-standing", 7: "resting-lying",
    8: "drinking", 9: "grooming", 10: "other", 11: "hidden", 12: "running",
}
# Comportements gardés (utiles + assez représentés) → classes 0..K-1
KEEP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 12: 7}
CLASS_NAMES = [BEH_MAP[b] for b in sorted(KEEP, key=lambda b: KEEP[b])]

CLIP_LEN, SIZE = 16, 112
# Normalisation Kinetics attendue par R(2+1)D-18 (torchvision)
MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
_BEH = re.compile(r"_beh(\d+)_")


def list_clips(root: str):
    items = []
    for d in sorted(glob.glob(os.path.join(root, "raw_frames", "*"))):
        m = _BEH.search(os.path.basename(d))
        if not m:
            continue
        behn = int(m.group(1))
        if behn not in KEEP:
            continue
        frames = sorted(glob.glob(os.path.join(d, "*.jpg")))
        if len(frames) >= CLIP_LEN:
            items.append((frames, KEEP[behn]))
    return items


class CVBClips(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        frames, label = self.items[i]
        idx = np.linspace(0, len(frames) - 1, CLIP_LEN).astype(int)
        buf = []
        for j in idx:
            img = cv2.imread(frames[j])
            img = cv2.cvtColor(cv2.resize(img, (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
            buf.append(img)
        x = torch.from_numpy(np.stack(buf)).float().div_(255.0)  # (T,H,W,3)
        x = x.permute(3, 0, 1, 2).contiguous()                    # (3,T,H,W)
        x = (x - MEAN) / STD
        return x, label


def build_model(num_classes: int) -> nn.Module:
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cvb", required=True, help="racine data/ contenant raw_frames/")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", default="behavior_video.pt")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    items = list_clips(args.cvb)
    print(f"{len(items)} clips · {len(CLASS_NAMES)} classes : {CLASS_NAMES} · device={device}")
    if not items:
        raise SystemExit("Aucun clip trouvé — vérifie --cvb (doit contenir raw_frames/).")

    n_val = max(1, int(0.15 * len(items)))
    ds = CVBClips(items)
    tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4)
    vl = DataLoader(va, batch_size=args.batch, num_workers=4)

    model = build_model(len(CLASS_NAMES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); lossf(model(x), y).backward(); opt.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item(); total += y.numel()
        print(f"epoch {ep+1}/{args.epochs}  val_acc={correct/max(1,total):.3f}")

    torch.save({
        "state_dict": model.cpu().state_dict(),
        "class_names": CLASS_NAMES,
        "clip_len": CLIP_LEN, "size": SIZE,
        "mean": MEAN, "std": STD, "arch": "r2plus1d_18",
    }, args.out)
    print(f"Sauvé : {args.out}")


if __name__ == "__main__":
    main()
