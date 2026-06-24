# Boeuf Tracker — YOLOv11 + DINOv2 + Re-ID

Système de reconnaissance individuelle de bovins:
- **YOLOv11** (Ultralytics) → détection + tracking intra-vidéo
- **DINOv2** (Meta) → extraction d'empreinte visuelle (embedding)
- **Cosine similarity** → reconnaissance cross-vidéo
- Base persistante (pickle) → les bovins identifiés hier sont reconnus aujourd'hui

## Installation

```powershell
cd C:\Users\Ismae\.minimax-agent\projects\boeuf-tracker
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Si tu as un GPU NVIDIA: installe CUDA + cuDNN avant pip install.
> Sinon ça tourne en CPU (~5-10 FPS avec yolo11n).

## Utilisation

### Mode interactif (menu)
```powershell
python main.py
```
→ Choisir entre une vidéo ou la webcam.

### Ligne de commande
```powershell
python main.py --source 0                    # webcam
python main.py --source videos\boeuf1.mp4    # fichier vidéo
python main.py --source rtsp://192.168.1.10:8554/stream  # flux (futur Pi)
```

### Options
| Flag | Défaut | Description |
|---|---|---|
| `--source` | menu | Chemin vidéo, index webcam ou URL |
| `--threshold` | 0.55 | Seuil cosine pour Re-ID (0.4 permissif, 0.7 strict) |
| `--db` | `cattle_db.pkl` | Fichier de base d'embeddings |
| `--yolo-model` | `yolo11n.pt` | Modèle YOLO (`yolo11s/m/l/x.pt` pour plus de précision) |
| `--dino-model` | `facebook/dinov2-small` | Modèle DINOv2 (`-base` ou `-large` plus précis) |
| `--conf` | 0.4 | Confiance minimale YOLO |

### Contrôles en cours d'exécution
- **`q`** → quitter (sauvegarde automatique)
- **`s`** → screenshot
- **`n`** → renommer le dernier bovin détecté (utile pour donner un vrai nom)

## Workflow typique

1. **Premier lancement (vidéo 1)**
   - Lance avec une vidéo de tes bœufs
   - Le système détecte et nomme automatiquement `Boeuf_001`, `Boeuf_002`…
   - Pendant la vidéo, appuie sur `n` pour donner un vrai nom (`Marguerite`, `Carbone`…)
   - Quitte avec `q` → la base est sauvegardée dans `cattle_db.pkl`

2. **Lancement suivant (vidéo 2 ou webcam)**
   - Les bœufs déjà en base sont **automatiquement reconnus**
   - Tu vois le même nom réapparaître → Re-ID cross-vidéo fonctionnel

3. **Pour augmenter la précision**
   - Fine-tune YOLO sur tes propres bœufs (Roboflow → Colab)
   - Fine-tune DINOv2 (triplet loss) sur tes paires positives

## Fichiers

```
boeuf-tracker/
├── main.py         # point d'entrée, boucle vidéo
├── detector.py     # YOLOv11 wrapper
├── reid.py         # DINOv2 embeddings
├── database.py     # stockage pickle
├── requirements.txt
└── cattle_db.pkl   # (généré après 1er run)
```

## Prochaine étape: Raspberry Pi

Quand ce pipeline marche bien en local, le Pi enverra le flux via:
- WebSocket (MJPEG base64 → server)
- ou RTSP (Pi + MediaMTX)

Le serveur GPU (ton PC ou un Jetson Orin) fera le même `main.py` mais en mode "serveur", écoutant les frames entrantes.