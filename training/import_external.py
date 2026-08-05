"""
training/import_external.py
---------------------------
Ingère des datasets EXTERNES (Roboflow, Kaggle, académiques…) et les convertit
au format attendu par train_posture.py : imagettes rangées par posture +
embeddings MegaDescriptor.

But : apprendre la posture sur un LARGE éventail de bovins (races, fermes,
lumières) et pas seulement sur tes vidéos, pour que le modèle généralise.

Ce que le script gère
---------------------
- Format DÉTECTION YOLO : `images/` + `labels/*.txt` (classe cx cy w h) +
  un `data.yaml` ou `classes.txt` donnant les noms de classes. On découpe
  chaque boîte et on hérite de sa classe.
- Format CLASSIFICATION par dossiers : `<classe>/*.jpg`. L'image entière hérite
  du nom de dossier.
- Plusieurs datasets à la fois : un sous-dossier par dataset sous --root.

Harmonisation des classes
-------------------------
Chaque dataset nomme ses classes différemment. CLASS_MAP ramène tout ça à trois
postures. Les classes qui ne sont PAS une posture (drinking, rumination,
estrus, générique « cow »…) sont ignorées : elles n'apprennent rien à un
classifieur de posture.

Usage
-----
    # 1. Télécharge tes datasets (ton compte) sous un dossier, un par sous-dossier :
    #    external/counting-cattle/ , external/cow-behavior/ , ...
    # 2. Ingère tout :
    python training/import_external.py --root external --out training/dataset

    # options :
    #   --max-per-class N   plafonne par posture (évite qu'un gros dataset écrase les autres)
    #   --stride K          n'ingère qu'une image sur K (sous-échantillonnage)
    #   --append            ajoute au dataset existant au lieu de le recréer

Les imagettes atterrissent dans les MÊMES dossiers que collect_dataset.py :
tu peux donc MÉLANGER images externes + tes vidéos, puis lancer train_posture.py
sans rien changer.
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from reid import CattleReID  # noqa: E402
from processor import _mlx_available, resolve_device  # noqa: E402

LABELS = ("couché", "debout", "pature", "autre")

# Mots-clés (en minuscules) -> posture. On teste par inclusion de sous-chaîne,
# du plus spécifique au plus général. L'ordre compte : "resting-lying" doit
# tomber sur couché avant que "standing" ne matche autre chose.
CLASS_KEYWORDS = [
    # couché
    ("couché", ["lying", "lie", "resting-lying", "ruminating-lying", "laying",
                 "couch", "recumbent", "down-cow", "cow-lying", "sleep"]),
    # pâture (mufle au sol : broute / mange)
    ("pature", ["graz", "forag", "feed", "eating", "browsing", "pature", "pâtur"]),
    # debout (posture verticale, y compris marche : ce n'est pas la vitesse
    # qu'on apprend ici, seulement la posture)
    ("debout", ["standing", "stand", "resting-standing", "ruminating-standing",
                 "walk", "upright", "debout", "active"]),
]

# Classes explicitement ignorées (pas une posture, ou trop générique pour
# étiqueter une posture de façon fiable).
IGNORE_KEYWORDS = ["drink", "rumination", "estrus", "mount", "fight", "lick",
                   "groom", "search", "other", "hidden", "non-active",
                   "unknown", "background"]


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def map_class(raw_name: str, overrides: dict[str, str] | None = None) -> str | None:
    """Ramène un nom de classe brut à couché/debout/pature, ou None si à ignorer.

    `overrides` (issu de --map) PRIME sur tout : c'est le seul moyen de trancher
    les classes ambiguës propres à un dataset (ex. 'resting' qui, selon les
    images, désigne un animal couché ou simplement debout au repos).
    """
    n = _norm(raw_name)
    if overrides and n in overrides:
        posture = overrides[n]
        return None if posture in ("ignore", "autre", "skip") else posture
    for kw in IGNORE_KEYWORDS:
        if kw in n:
            return None
    for posture, keys in CLASS_KEYWORDS:
        if any(k in n for k in keys):
            return posture
    # Générique "cow"/"cattle" seul = aucune info de posture -> ignore.
    return None


# ── Lecture des noms de classes d'un dataset YOLO ──────────────────────────
def _yolo_class_names(ds_dir: str) -> list[str] | None:
    # data.yaml (Roboflow) : names: ['a','b',...]  ou  names:\n  0: a\n  1: b
    for cand in ("data.yaml", "data.yml", "classes.txt", "obj.names"):
        p = os.path.join(ds_dir, cand)
        if not os.path.exists(p):
            continue
        if cand.endswith((".yaml", ".yml")):
            try:
                import yaml
                names = yaml.safe_load(open(p)).get("names")
                if isinstance(names, dict):
                    return [names[k] for k in sorted(names)]
                if isinstance(names, list):
                    return names
            except Exception:
                # Parse minimal si pyyaml absent : ligne "names: [a, b, c]"
                for line in open(p):
                    line = line.strip()
                    if line.startswith("names:") and "[" in line:
                        inside = line.split("[", 1)[1].rsplit("]", 1)[0]
                        return [x.strip().strip("'\"") for x in inside.split(",")]
        else:  # classes.txt / obj.names : une classe par ligne
            return [l.strip() for l in open(p) if l.strip()]
    return None


def _iter_yolo(ds_dir: str, names: list[str]):
    """Génère (image_path, [(raw_class, x1,y1,x2,y2), ...]) pour un dataset YOLO."""
    # Cherche les paires images/labels dans les splits usuels.
    for split in ("", "train", "valid", "test"):
        img_dir = os.path.join(ds_dir, split, "images") if split else os.path.join(ds_dir, "images")
        lbl_dir = os.path.join(ds_dir, split, "labels") if split else os.path.join(ds_dir, "labels")
        if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
            continue
        for fn in os.listdir(img_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fn)[0]
            lbl = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lbl):
                continue
            img_path = os.path.join(img_dir, fn)
            im = cv2.imread(img_path)
            if im is None:
                continue
            H, W = im.shape[:2]
            boxes = []
            for line in open(lbl):
                parts = line.split()
                if len(parts) < 5:
                    continue
                ci = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * W); y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W); y2 = int((cy + bh / 2) * H)
                raw = names[ci] if 0 <= ci < len(names) else str(ci)
                boxes.append((raw, max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
            if boxes:
                yield img_path, im, boxes


def _cvb_paths(ds_dir: str):
    """Détecte un dataset CVB (format AVA) et retourne ses chemins, sinon None.

    Structure CVB : <ds>/data/cvb_in_ava_format/{ava_*.csv, behaviour_list.pbtx}
    + <ds>/data/raw_frames/<clip>/img_00001.jpg …
    On tolère que `data/` soit présent ou non (selon comment le dossier a été
    déposé sous --root).
    """
    for base in (ds_dir, os.path.join(ds_dir, "data")):
        ava = os.path.join(base, "cvb_in_ava_format")
        frames = os.path.join(base, "raw_frames")
        pbtx = os.path.join(ava, "behaviour_list.pbtx")
        if os.path.isdir(ava) and os.path.isdir(frames) and os.path.exists(pbtx):
            return {"ava": ava, "frames": frames, "pbtx": pbtx}
    return None


def _parse_pbtx(pbtx_path: str) -> dict[int, str]:
    """behaviour_list.pbtx -> {label_id: name}."""
    import re
    txt = open(pbtx_path).read()
    ids = re.findall(r'label_id:\s*(\d+)', txt)
    names = re.findall(r'name:\s*"([^"]+)"', txt)
    return {int(i): n for i, n in zip(ids, names)}


def _iter_ava(paths: dict):
    """Génère (frame_path, image, [(behavior_name, x1,y1,x2,y2), ...]) pour CVB.

    Chaque ligne AVA : clip, seconde, x1,y1,x2,y2 (normalisés), behavior_id,
    entity_id. On regroupe par (clip, seconde) pour ne lire chaque image
    qu'une fois, et on vise la frame du MILIEU de la seconde annotée
    (posture stable sur une seconde). L'ordre est mélangé pour répartir la
    contribution entre clips plutôt que d'épuiser le quota sur les premiers.
    """
    import csv
    import random
    id2name = _parse_pbtx(paths["pbtx"])
    frames_root = paths["frames"]

    # Lecture des lignes des deux splits (train + val).
    groups: dict[tuple, list] = {}
    for split in ("ava_train_set.csv", "ava_val_set.csv"):
        p = os.path.join(paths["ava"], split)
        if not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 8:
                    continue
                clip, ts = row[0], row[1]
                x1, y1, x2, y2 = map(float, row[2:6])
                name = id2name.get(int(row[6]), str(row[6]))
                groups.setdefault((clip, ts), []).append((name, x1, y1, x2, y2))

    # Nombre de frames par clip (pour caler seconde -> image). Certains clips
    # peuvent être incomplets si le téléchargement n'est pas fini : on saute.
    frame_count: dict[str, int] = {}
    max_ts: dict[str, int] = {}
    for (clip, ts) in groups:
        max_ts[clip] = max(max_ts.get(clip, 0), int(ts))
    keys = list(groups.keys())
    random.Random(0).shuffle(keys)

    for clip, ts in keys:
        cdir = os.path.join(frames_root, clip)
        if clip not in frame_count:
            frame_count[clip] = (len(os.listdir(cdir))
                                 if os.path.isdir(cdir) else 0)
        n = frame_count[clip]
        if n == 0:
            continue
        fpts = n / max(max_ts[clip], 1)          # frames par seconde annotée
        fidx = int(round((int(ts) - 0.5) * fpts))  # milieu de la seconde
        fidx = min(max(fidx, 1), n)
        img_path = os.path.join(cdir, f"img_{fidx:05d}.jpg")
        if not os.path.exists(img_path):
            continue
        im = cv2.imread(img_path)
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = [(name, int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H))
                 for (name, x1, y1, x2, y2) in groups[(clip, ts)]]
        yield img_path, im, boxes


def _iter_folders(ds_dir: str):
    """Génère (image_path, image, [(raw_class, full_box)]) pour un dataset
    classification (un dossier par classe)."""
    for entry in os.listdir(ds_dir):
        cdir = os.path.join(ds_dir, entry)
        if not os.path.isdir(cdir):
            continue
        for fn in os.listdir(cdir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            p = os.path.join(cdir, fn)
            im = cv2.imread(p)
            if im is None:
                continue
            H, W = im.shape[:2]
            yield p, im, [(entry, 0, 0, W, H)]


def ingest(root, out_dir, max_per_class, stride, append, overrides=None,
           per_source_cap=0, batch=64):
    os.makedirs(out_dir, exist_ok=True)
    for lab in LABELS:
        os.makedirs(os.path.join(out_dir, lab), exist_ok=True)

    reid_dev = "mps" if _mlx_available() else resolve_device("auto")
    reid = CattleReID(device=reid_dev)

    emb_path = os.path.join(out_dir, "embeddings.pkl")
    embeddings: dict[str, np.ndarray] = {}
    if append and os.path.exists(emb_path):
        embeddings = pickle.load(open(emb_path, "rb"))

    per_class = {"couché": 0, "debout": 0, "pature": 0}
    skipped = 0
    datasets = [d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))]
    if not datasets:
        # --root pointe peut-être directement sur UN dataset.
        datasets = ["."]

    # Buffer de crops en attente d'embedding. On accumule jusqu'a `batch` puis
    # on calcule tous les embeddings EN UN SEUL forward GPU (get_embedding_batch).
    # C'est le gros levier de vitesse : un forward de 64 imagettes est bien plus
    # rapide que 64 forwards isoles (le GPU reste sature au lieu d'attendre).
    pending: list[tuple[np.ndarray, str, str]] = []  # (crop, label, fname)
    counters = {"n": 0}

    def flush():
        if not pending:
            return
        crops = [c for c, _, _ in pending]
        embs = reid.get_embedding_batch(crops)
        for (crop, label, fname), emb in zip(pending, embs):
            if emb is None or emb.shape[0] != reid.TOTAL_DIM:
                continue
            cv2.imwrite(os.path.join(out_dir, label, fname), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            embeddings[fname] = np.asarray(emb, dtype=np.float32)
            counters["n"] += 1
        pending.clear()

    for ds in datasets:
        ds_dir = os.path.join(root, ds) if ds != "." else root
        tag = os.path.basename(ds_dir.rstrip("/")) or "ds"
        cvb = _cvb_paths(ds_dir)
        names = None if cvb else _yolo_class_names(ds_dir)
        if cvb:
            source, kind = _iter_ava(cvb), "CVB/AVA"
        elif names:
            source, kind = _iter_yolo(ds_dir, names), "YOLO"
        else:
            source, kind = _iter_folders(ds_dir), "dossiers"
        counters["n"] = 0
        # Quota par (dataset, posture) : garantit que CHAQUE source contribue,
        # au lieu de laisser les 2 premiers datasets saturer les plafonds
        # globaux et affamer les suivants (perte de diversite de races).
        per_source = {"couché": 0, "debout": 0, "pature": 0}
        for idx, (img_path, im, boxes) in enumerate(source):
            if idx % stride != 0:
                continue
            for bi, (raw, x1, y1, x2, y2) in enumerate(boxes):
                label = map_class(raw, overrides)
                if label is None:
                    skipped += 1
                    continue
                if per_class[label] >= max_per_class:
                    continue
                if per_source_cap and per_source[label] >= per_source_cap:
                    continue
                if (x2 - x1) < 24 or (y2 - y1) < 24:
                    continue
                crop = im[y1:y2, x1:x2].copy()  # copie: im sera reutilise/libere
                fname = f"ext_{tag}_{idx}_{bi}.jpg"
                pending.append((crop, label, fname))
                per_class[label] += 1
                per_source[label] += 1
                if len(pending) >= batch:
                    flush()
        flush()  # vide le reste du dataset avant de passer au suivant
        print(f"[{tag}] {counters['n']} imagettes ({kind})")

    pickle.dump(embeddings, open(emb_path, "wb"))
    print(f"\nTotal par posture (imagettes externes) :")
    for k, v in per_class.items():
        print(f"  {k:8s}: {v}")
    print(f"Classes ignorées (non-posture) : {skipped} boîtes")
    print("\nÉtape suivante : vérifie un échantillon dans chaque dossier "
          "(l'harmonisation peut se tromper), puis lance train_posture.py")


def main():
    ap = argparse.ArgumentParser(description="Ingestion de datasets externes.")
    ap.add_argument("--root", required=True,
                    help="dossier contenant un sous-dossier par dataset")
    ap.add_argument("--out", default="training/dataset")
    # Plafond GLOBAL par posture. Volontairement très haut par défaut : c'est
    # --per-source qui règle l'équilibre entre datasets. Un défaut bas (ex.
    # 4000) se remplit avec les 2 premiers datasets et affame tous les suivants
    # (les derniers traités n'ingèrent presque rien) — piège vécu avec CVB.
    ap.add_argument("--max-per-class", type=int, default=1_000_000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--append", action="store_true",
                    help="ajoute au dataset existant (ex: tes vidéos déjà collectées)")
    ap.add_argument("--map", nargs="*", default=[], metavar="CLASSE=POSTURE",
                    help="force le mapping d'une classe ambiguë. POSTURE ∈ "
                         "{couché, debout, pature, ignore}. Ex: "
                         "--map resting=couché \"resting bout=couché\" walking=ignore")
    ap.add_argument("--per-source", type=int, default=0,
                    help="plafond par dataset ET par posture, pour que chaque "
                         "source contribue (diversite). Ex: 1500")
    ap.add_argument("--batch", type=int, default=64,
                    help="nombre d'imagettes embeddees en un seul forward GPU "
                         "(vitesse). 64 est un bon compromis; monte si tu as de "
                         "la marge memoire.")
    args = ap.parse_args()

    overrides: dict[str, str] = {}
    for item in args.map:
        if "=" not in item:
            raise SystemExit(f"--map invalide: '{item}' (attendu CLASSE=POSTURE)")
        raw, posture = item.rsplit("=", 1)
        posture = posture.strip().lower()
        valid = {"couché", "couche", "debout", "pature", "pâture",
                 "ignore", "autre", "skip"}
        if posture not in valid:
            raise SystemExit(f"--map: posture '{posture}' inconnue "
                             f"(choix: couché, debout, pature, ignore)")
        # normalise l'accent des postures pour coller aux dossiers LABELS
        posture = {"couche": "couché", "pâture": "pature"}.get(posture, posture)
        overrides[_norm(raw)] = posture

    ingest(args.root, args.out, args.max_per_class, args.stride, args.append,
           overrides, args.per_source, args.batch)


if __name__ == "__main__":
    main()
