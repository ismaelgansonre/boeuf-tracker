"""
colab_x3d_cvb.py
================
Pipeline COLAB : entraîne un X3D (comportement vidéo) sur le dataset CVB,
puis exporte un checkpoint réimportable dans TON code Python (behavior_x3d.py).

À exécuter dans Google Colab (Runtime = GPU). Les cellules sont délimitées par
`# %%`. Tu peux aussi l'uploader tel quel et faire "Run all".

Sortie finale :
  - x3d_cvb.pt        : poids + méta (labels, config) → à copier dans training/
  - un mode d'emploi pour l'inférence dans ton pipeline (voir behavior_x3d.py)

Pourquoi X3D : le plus léger des modèles vidéo SOTA, pré-entraîné Kinetics-400,
on n'affine que la tête sur les comportements CVB. Convertible CoreML ensuite.
"""

# %% [1] Installation
# (dans Colab)
#   !pip -q install torch torchvision pytorchvideo av
# torch/torchvision sont déjà présents dans Colab ; pytorchvideo apporte X3D.

# %% [2] Récupérer CVB via S3 (rclone) — DIRECTEMENT dans Google Drive
#
# CSIRO expose le dataset en S3 (accès public en lecture). rclone le copie sur
# les serveurs Google, dans ton Drive → téléchargé UNE fois, relu à chaque
# session Colab sans retélécharger.
#
#   from google.colab import drive
#   drive.mount('/content/drive')
#   !curl -s https://rclone.org/install.sh | sudo bash    # rclone récent
#
#   # Copie S3 -> Drive (clés d'accès PUBLIQUES du dataset CSIRO)
#   !rclone -P --s3-provider Other --s3-endpoint https://s3.data.csiro.au \
#       --s3-access-key-id HZ7ZM59VGWJ3P5MM1LO7 \
#       --s3-secret-access-key HV9n5LmgwWtmNlXRbmEFERsGe7x43HbLCQGHeIhc \
#       --transfers 8 --multi-thread-cutoff 250M --multi-thread-streams 4 \
#       copy :s3:dapprd/000058916v001/ /content/drive/MyDrive/cvb/58916v001
#
# ⚠️ Drive (FUSE) est lent pour 16k+ petits fichiers. Deux options :
#   (A) Copie directe vers Drive (ci-dessus) : simple, un peu lent, mais 1 seule fois.
#   (B) Plus rapide à RÉUTILISER : rclone vers /content (disque local Colab, rapide),
#       puis tar → Drive :
#         !rclone -P ... copy :s3:dapprd/000058916v001/ /content/cvb
#         !tar -C /content -cf /content/drive/MyDrive/cvb.tar cvb
#       et au début de chaque session : !tar -C /content -xf /content/drive/MyDrive/cvb.tar
#
# Pointe ensuite la racine "data" (là où se trouve raw_frames/) :
CVB_ROOT = "/content/drive/MyDrive/cvb/58916v001/data"   # ajuste si l'arbo diffère

# %% [3] Construire le dataset de clips (label = behN du nom de dossier)
import os, re, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

# Mapping behN (nom de dossier) → nom lisible. Aligné sur behaviour_list.pbtx.
BEH_MAP = {
    1: "none", 2: "grazing", 3: "walking", 4: "ruminating-standing",
    5: "ruminating-lying", 6: "resting-standing", 7: "resting-lying",
    8: "drinking", 9: "grooming", 10: "other", 11: "hidden", 12: "running",
}
# On ne garde que les comportements utiles et bien représentés (à ajuster).
KEEP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 12: 7}  # behN -> classe 0..K
CLASS_NAMES = [BEH_MAP[b] for b in sorted(KEEP, key=lambda b: KEEP[b])]

CLIP_LEN, SIZE = 16, 182
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)
_BEH = re.compile(r"_beh(\d+)_")


def list_clips(root):
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
        # échantillonnage temporel uniforme de CLIP_LEN frames
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


# %% [4] Modèle X3D + entraînement
import torch.nn as nn


def build_x3d(num_classes):
    model = torch.hub.load("facebookresearch/pytorchvideo", "x3d_m", pretrained=True)
    proj = model.blocks[-1].proj
    model.blocks[-1].proj = nn.Linear(proj.in_features, num_classes)
    return model


def train(root, epochs=15, batch=8, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = list_clips(root)
    print(f"{len(items)} clips, {len(CLASS_NAMES)} classes : {CLASS_NAMES}")
    n_val = max(1, int(0.15 * len(items)))
    ds = CVBClips(items)
    tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    tl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=2)
    vl = DataLoader(va, batch_size=batch, num_workers=2)

    model = build_x3d(len(CLASS_NAMES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); lossf(model(x), y).backward(); opt.step()
        # validation
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item(); total += y.numel()
        print(f"epoch {ep+1}/{epochs}  val_acc={correct/max(1,total):.3f}")
    return model.cpu().eval()


# %% [5] Entraîner + exporter
model = train(CVB_ROOT, epochs=15)

torch.save({
    "state_dict": model.state_dict(),
    "class_names": CLASS_NAMES,
    "clip_len": CLIP_LEN, "size": SIZE,
    "mean": MEAN, "std": STD,
    "arch": "x3d_m",
}, "x3d_cvb.pt")
print("Sauvé : x3d_cvb.pt  → copie-le dans ton dossier training/")

# %% [6] (Optionnel) Export CoreML pour l'ANE, si tu veux un jour le natif
# import coremltools as ct
# ex = torch.rand(1, 3, CLIP_LEN, SIZE, SIZE)
# ts = torch.jit.trace(model, ex)
# ct.convert(ts, inputs=[ct.TensorType(name="clip", shape=ex.shape)],
#            compute_precision=ct.precision.FLOAT16,
#            minimum_deployment_target=ct.target.macOS14
#            ).save("X3DBehavior.mlpackage")
