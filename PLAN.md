# PLAN — Boeuf Tracker (reconnaissance + comportement des bovins)

Dernière mise à jour : 2026-08-04

## 1. Objectif

Reconnaître **individuellement** chaque bœuf sur vidéo (le nommer, le sauvegarder),
le **compter**, et déterminer son **comportement** (immobile, marche, pâture,
rumination, etc.). Cible : ~85 % de bonne détection de comportement. La ReID
(identité) est le must-have ; le comportement est le bonus.

## 2. Décision stratégique : Python (pas Swift)

- Le projet Python existant marche déjà bien (YOLO+ByteTrack+MegaDescriptor+posture).
- Réécrire en Swift = réinventer ce qu'Ultralytics donne gratuitement (masques,
  letterbox, NMS, imgsz réglable, tracking). Trop coûteux.
- **L'ANE est accessible DEPUIS Python** via coremltools / Ultralytics-CoreML
  (`YOLO("modele.mlpackage")` tourne sur le Neural Engine, macOS uniquement).
  Vérifié sur la doc Apple Core ML Tools.
- => On reste en Python. Le natif Swift est une étape d'empaquetage OPTIONNELLE,
  plus tard. (Le prototype Swift reste dans /Users/ovoxost/Desktop/GIT/BoeufTracker
  à titre d'expérience — non prioritaire.)

## 3. Règle matérielle (ANE vs GPU)

| Phase | Matériel |
|-------|----------|
| Entraînement (tous modèles) | GPU (MPS/CUDA) — jamais l'ANE |
| Inférence YOLO26 / MegaDescriptor (2D) | ANE ✅ (via CoreML) |
| Inférence comportement vidéo (conv 3D) | surtout GPU (l'ANE gère mal la 3D) |

## 4. Architecture cible (pipeline Python)

```
Vidéo
 → YOLO26-seg (ANE)        détection + segmentation + comptage
 → ByteTrack               suivi, track_id stable
 → MegaDescriptor (ANE)    empreinte → identité → NOM du bœuf (ReID few-shot)
 → base de données         troupeau sauvegardé, historique (reset possible)
 → posture + vitesse       immobile / marche / pâture (règles, posture.py)
 → R(2+1)D-18 (GPU)        comportement fin : grazing, ruminating, walking, ...
```

## 5. Modèles

| Modèle | Rôle | Entraîné ? | Où |
|--------|------|-----------|-----|
| YOLO26-seg | détection/comptage | non (COCO 'cow') | ANE |
| MegaDescriptor | identité (ReID) | non (pré-entraîné) | ANE |
| R(2+1)D-18 | comportement vidéo | **OUI, sur CVB** | GPU |

Note : on visait X3D (pytorchvideo) mais il casse avec torch 2.13.
Bascule sur **R(2+1)D-18 (torchvision, maintenu, pré-entraîné Kinetics-400)**.
Même principe (CNN 3D vidéo), plus robuste.

## 6. Données

- `training/dataset/` : images triées Kaggle/Roboflow (couché/debout/pature).
  Les crops `IMG_*` issus des vidéos ont été RETIRÉS (mal triés) →
  `training/_img_excluded/`. Le dataset ne garde que les images externes.
- `external/cvb_full/58916v001/data/` : **CVB** (Cattle Visual Behaviours, CSIRO),
  13,8 Go, 502 clips vidéo annotés, 11 comportements. Téléchargé via rclone S3.
  Endpoint : s3.data.csiro.au, bucket dapprd/000058916v001.
- 8 comportements retenus pour l'entraînement : grazing, walking,
  ruminating-standing, ruminating-lying, resting-standing, resting-lying,
  drinking, running.

## 7. Fichiers clés (repo Python boeuf-tracker)

- `training/train_behavior_video.py` — entraîne R(2+1)D-18 sur CVB (local, MPS)
- `behavior_video.py` — inférence comportement, classe `VideoBehavior`
  (buffer 16 frames par track → comportement), à brancher dans processor.py
- `training/colab_x3d_cvb.py` — variante Colab (obsolète : Colab free banni)
- `detector.py`, `reid.py`, `posture.py`, `behavior.py`, `processor.py` — existants

## 8. État actuel

- [x] CVB téléchargé (13,8 Go, 502 clips)
- [x] Crash MPS (conv 3D) résolu : `PYTORCH_ENABLE_MPS_FALLBACK=1` + batch 2
- [ ] Entraînement R(2+1)D-18 EN COURS — epoch 1/15 : val_acc 0.619
- [ ] `behavior_video.pt` (produit à la fin)
- [ ] Brancher `VideoBehavior` dans `processor.py`

## 9. Commandes utiles

Entraînement (relance manuelle) :
```
cd /Users/ovoxost/Desktop/GIT/boeuf-tracker
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python training/train_behavior_video.py \
  --cvb external/cvb_full/58916v001/data --epochs 15 --batch 2
```

Suivi des logs :
```
tail -f /tmp/train_retry.log | grep --line-buffered val_acc
```

## 10. Prochaines étapes

1. Attendre la fin des 15 epochs → lire le score final par comportement.
2. Si une classe décroche (ex. drinking rare) → rééquilibrage / plus d'epochs.
3. Brancher `VideoBehavior` dans `processor.py` (là où track_id + crop existent).
4. (Optionnel) Utiliser YOLO26/MegaDescriptor en CoreML depuis Python pour l'ANE.
5. Si MPS retombe instable → repli Kaggle (GPU CUDA gratuit, fiable pour la 3D).

## 11. Repli si l'entraînement local échoue

Kaggle (GPU T4 gratuit, 30h/sem, CUDA fiable pour conv 3D) :
- tirer CVB depuis le S3 CSIRO (mêmes clés rclone)
- même script d'entraînement
- récupérer seulement `behavior_video.pt` (quelques Mo)
