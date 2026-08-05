"""
training/train_x3d.py
---------------------
Étape 2 de la méthode X3D : fine-tune X3D (modèle vidéo 3D CNN) sur les clips
denses produits par build_clips.py, puis export CoreML float16 pour l'ANE.

POURQUOI X3D
------------
X3D-M est le plus LÉGER des modèles vidéo SOTA (conçu pour l'efficience) →
le plus réaliste à faire tourner sur Apple Silicon. Pré-entraîné Kinetics-400,
on ne ré-apprend que la tête de classification (couché/debout/pature).

ENTRÉE MODÈLE : (B, 3, T=16, 182, 182), normalisée Kinetics.

Usage
-----
    pip install torch torchvision pytorchvideo coremltools
    python training/train_x3d.py --clips training/clips --epochs 15
    # → x3d_behavior.mlpackage  (à glisser dans le projet Xcode)

NOTE ANE : les convolutions 3D d'X3D passent en partie sur GPU Apple plutôt que
100% ANE. Sur Apple Silicon ça reste temps réel ; vérifier la répartition dans
le Performance report d'Xcode.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

LABELS = ("couché", "debout", "pature")
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}

# Normalisation Kinetics-400 (attendue par X3D pré-entraîné)
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)


class ClipDataset(Dataset):
    """Charge les .npy (T,H,W,3) uint8 → tenseur (3,T,H,W) normalisé."""

    def __init__(self, clips_dir: str):
        self.dir = clips_dir
        self.items: list[tuple[str, int]] = []
        with open(os.path.join(clips_dir, "index.csv")) as f:
            for row in csv.DictReader(f):
                self.items.append((row["path"], LABEL2IDX[row["label"]]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        rel, label = self.items[i]
        clip = np.load(os.path.join(self.dir, rel))          # (T,H,W,3) uint8
        x = torch.from_numpy(clip).float().div_(255.0)       # (T,H,W,3)
        x = x.permute(3, 0, 1, 2).contiguous()               # (3,T,H,W)
        x = (x - MEAN) / STD
        return x, label


def build_model(num_classes: int) -> nn.Module:
    """X3D-M pré-entraîné, tête remplacée par nos classes."""
    model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)
    # remplacer la couche de projection finale
    proj = model.blocks[-1].proj
    model.blocks[-1].proj = nn.Linear(proj.in_features, num_classes)
    return model


def train(args) -> nn.Module:
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    ds = ClipDataset(args.clips)
    n_val = max(1, int(0.15 * len(ds)))
    train_ds, val_ds = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    vl = DataLoader(val_ds, batch_size=args.batch, num_workers=4)

    model = build_model(len(LABELS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item()
                total += y.numel()
        print(f"epoch {epoch+1}/{args.epochs}  val_acc={correct/max(1,total):.3f}")

    return model.cpu().eval()


def export_coreml(model: nn.Module, clip_len: int, size: int, out: str) -> None:
    import coremltools as ct
    example = torch.rand(1, 3, clip_len, size, size)
    traced = torch.jit.trace(model, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="clip", shape=example.shape)],
        compute_precision=ct.precision.FLOAT16,   # ANE-friendly
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    # attacher les labels comme métadonnées lisibles côté Swift
    mlmodel.user_defined_metadata["labels"] = ",".join(LABELS)
    mlmodel.save(out)
    print(f"[export] OK → {out} (float16)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clips", default="training/clips")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--clip-len", type=int, default=16)
    p.add_argument("--size", type=int, default=182)
    p.add_argument("--out", default="x3d_behavior.mlpackage")
    args = p.parse_args()

    model = train(args)
    export_coreml(model, args.clip_len, args.size, args.out)


if __name__ == "__main__":
    main()
