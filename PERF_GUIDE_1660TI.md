# 🐮 Guide de perf — GTX 1660 Ti (6 GB VRAM, sm_75, Turing)

Ta carte a **1536 CUDA cores, 6 GB GDDR6, ~5.5 TFLOPS FP32, pas de Tensor Cores
dédiés** (les Turing ont du FP16 bridés). Donc on optimise pour **throughput
FP16 + imgsz modéré**, pas pour de la grosse précision.

## 🎯 Réglages recommandés (par profil)

| Profil | YOLO | Imgsz | embed_every | skip | conf | FPS attendus | Détection |
|---|---|---|---|---|---|---|---|
| **🚀 Max FPS (live webcam)** | `yolo11n-seg.pt` | 416 | 20 | 2 | 0.45 | **35-45** | OK si bœufs proches |
| **⚖️ Équilibré (recommandé)** | `yolo11s-seg.pt` | 640 | 10 | 1 | 0.40 | **20-28** | Très bonne |
| **🎯 Précision max (vidéo offline)** | `yolo11m-seg.pt` | 800 | 5 | 0 | 0.35 | **10-14** | Excellente |

⚠️ **Évite `yolo11l-seg.pt` / `yolo11x-seg.pt`** sur 1660 Ti : trop lourd,
FPS chute à <6 et tu n'as pas les Tensor Cores pour compenser.

## 🔧 Commandes prêtes à l'emploi

### Profil "équilibré" (recommandé pour toi)
```powershell
python app.py --yolo-model yolo11s-seg.pt --imgsz 640 --embed-every 10 --skip-frames 1 --conf 0.4
```

### Profil "max FPS" (webcam live, bétail proche)
```powershell
python app.py --yolo-model yolo11n-seg.pt --imgsz 416 --embed-every 20 --skip-frames 2 --conf 0.45
```

### Profil "précision" (vidéo enregistrée, fine-tuning)
```powershell
python app.py --yolo-model yolo11m-seg.pt --imgsz 800 --embed-every 5 --skip-frames 0 --conf 0.35
```

## ⚙️ Pourquoi ces choix

### YOLO + FP16
- `detector.py` active **déjà FP16** automatiquement sur CUDA (`half=True`).
- 1660 Ti = Turing : FP16 ~1.5x plus rapide que FP32 mais sans Tensor Cores
  pour aller plus loin. C'est déjà optimum.

### Imgsz
- 640 = sweet spot YOLOv11 (entraîné là-dessus).
- 416 = gain ~30% FPS, mais perd les bœufs <80px de haut dans la frame.
- 800 = +40% temps, gain marginal sur gros plans.

### skip-frames
- **1** = traiter 1 frame sur 2 = ~2x FPS mais tracking moins smooth.
- 0 = toutes les frames, le plus précis.
- 2 = 1 frame sur 3, fluide pour webcam mais lag pour ré-id.

### embed_every (DINOv2)
- DINOv2 small ≈ 90 MB, ~15 ms/inference sur 1660 Ti.
- Pour 3 bœufs simultanés : 3 × 15 ms = 45 ms toutes les N frames.
- À 25 FPS cible (40 ms/frame), il faut **N ≥ 2** sinon ça sature.

### conf
- 0.40 = détecte les bovins même partiellement cachés (recommandé plein air).
- 0.45 = moins de faux positifs (ombres, mottes de terre).
- < 0.35 = beaucoup de bruit, à éviter.

## 🧠 Optimisations supplémentaires (déjà en place)

✅ **FP16** activé par défaut
✅ **Skip frames** configurable
✅ **embed_every** configurable
✅ **JPEG quality 75** dans `processor.py:431` (bon compromis CPU/GPU)
✅ **EMA update** ne touche que les 4 derniers embeddings (`buf[-4:]`)
✅ **Validation dim** au boot : purge auto si modèle changé

## 🛠️ Si tu veux encore gratter (avancé)

### 1. Baisser la qualité JPEG → économie CPU encodage
Dans `processor.py:431` :
```python
ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])  # 75→60
```

### 2. Utiliser TensorRT (compile YOLO une fois, ~2x speedup)
```powershell
pip install ultralytics
yolo export model=yolo11s-seg.pt format=engine half=True imgsz=640 device=0
# Puis lancer avec --yolo-model yolo11s-seg.engine
```
⚠️ Nécessite CUDA + TensorRT installé (~2 GB).

### 3. Réduire embed_every pour les tracks stables
Tracks nouveaux → embed à chaque frame.
Tracks confirmés (>30 frames vus) → embed tous les 30 frames.
Pas implémenté, mais facile à ajouter (compare `frame_idx - first_seen_frame`).

## 🐛 Anti-double-comptage (déjà implémenté)

Avec les flags suivants, une vidéo en boucle ne crée plus de doublons :
```powershell
python app.py --max-updates 30 --loop-threshold 0.45 --loop-grace-frames 60
```

- `--max-updates 30` : l'embedding de référence est figé après 30 mises à jour
  (évite la dérive qui empêche la ré-id).
- `--loop-threshold 0.45` : seuil permissif juste après un rebobinage
  (ré-identifie malgré le changement de pose).
- `--loop-grace-frames 60` : durée de la période "permissive" après boucle.
