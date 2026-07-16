# PFE Individuel — Documentation Technique

## Boeuf Tracker : Système de Surveillance Bovine en Temps Réel

**Détection · Identification individuelle · Classification de race · Carte de densité**

---

| | |
|---|---|
| **Étudiant** | (anonymisé) |
| **Programme** | Baccalauréat en Génie Électrique |
| **Cours** | GEI1052 — Activités de synthèse |
| **Session** | Hiver 2026 |
| **Version** | 2.0 |
| **Date** | Juillet 2026 |

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture du système](#2-architecture-du-système)
3. [Le pipeline IA](#3-le-pipeline-ia)
4. [Identification individuelle (Re-ID)](#4-identification-individuelle-re-id)
5. [Classification de race](#5-classification-de-race)
6. [Carte de densité (Heatmap)](#6-carte-de-densité-heatmap)
7. [Interface utilisateur](#7-interface-utilisateur)
8. [Application desktop](#8-application-desktop)
9. [Performance et optimisation](#9-performance-et-optimisation)
10. [Installation et déploiement](#10-installation-et-déploiement)
11. [Fichiers du projet](#11-fichiers-du-projet)
12. [Glossaire](#12-glossaire)

---

## 1. Vue d'ensemble

### Ce que fait le système

Boeuf Tracker analyse un flux vidéo en temps réel et :

1. **Détecte** chaque bovin dans l'image (modèle YOLO)
2. **Identifie** chaque bovin individuellement — il reconnaît la même vache d'une frame à l'autre et lui attribue un nom unique
3. **Classifie** la race probable (Charolaise, Holstein, Salers...)
4. **Cartographie** les zones de regroupement (heatmap)
5. **Affiche** le tout dans une interface web avec dashboard analytics

### Métriques de performance

| Métrique | Valeur |
|----------|--------|
| FPS moyen | **23-26 FPS** |
| Latence détection YOLO | ~12 ms/frame |
| Latence Re-ID | ~5 ms (async) |
| Latence classification race | ~22 ms (1re détection) |
| Taille binaire desktop | 1.8 MB (Tauri) |
| Lignes de code | ~4700 Python + 800 JS + 120 Rust |

---

## 2. Architecture du système

Le système est composé de deux processus : une application desktop (Tauri) qui orchestre un worker Python (Flask) responsable de tout le traitement IA.

![Architecture du système](diagram_architecture.png)

### Les deux processus

| Processus | Rôle | Technologie |
|-----------|------|-------------|
| **Worker Python** (port 8100) | Traitement IA + API REST + UI statique | Flask, PyTorch, MLX |
| **App desktop** (Tauri) | Fenêtre native, gestion du cycle de vie | Rust, WKWebView |

Le worker Python fait tout : inférence IA, API REST, et sert les fichiers HTML/CSS/JS de l'interface. L'application Tauri n'est qu'une coquille qui ouvre une fenêtre native et démarre/arrête le worker Python.

---

## 3. Le pipeline IA

Voici ce qui se passe à chaque frame vidéo (~38 ms) :

![Pipeline de traitement](diagram_pipeline.png)

### Étape 1 — Détection YOLO

**YOLO** (You Only Look Once) détecte les objets en une seule passe. On utilise YOLOv26 avec le framework **MLX** d'Apple (accélération Metal GPU) sur Mac, ou YOLOv11 standard sur autres plateformes.

- **Entrée** : image 640×640
- **Sortie** : boîtes [x1,y1,x2,y2] + masque + confiance
- **Filtrage** : confiance ≥ 0.4, puis NMS (suppression des doublons)
- **Fichier** : `detector.py`

### Étape 2 — Tracking IoU

On associe les boîtes de la frame précédente à la frame actuelle via l'**Intersection over Union** (chevauchement). Le bovin #3 reste #3 tant qu'il est visible.

### Étape 3 — Extraction du crop

Découpe de la zone du bovin dans l'image originale : `crop = frame[y1:y2, x1:x2]`

---

## 4. Identification individuelle (Re-ID)

### Le problème

Le tracker IoU perd l'identifiant d'un bovin quand il sort du champ ou est occulté. Il faut reconnaître la **même vache** après une interruption.

### La solution : embeddings DINOv2

![Re-ID par embeddings](diagram_reid.png)

**DINOv2** (Meta AI, self-supervised) transforme chaque crop en un vecteur de 464 dimensions. Deux photos de la même vache donnent des vecteurs très similaires (cosinus proche de 1.0).

### Comparaison et base de données

Chaque bovin est stocké dans `cattle_db.pkl` avec son embedding moyen (mis à jour par moyenne mobile exponentielle). À chaque nouvelle détection, on compare l'embedding avec la base via cosine similarity :

- **sim ≥ 0.70** → bovin connu (nom conservé)
- **sim < 0.70** → nouveau bovin (nouveau nom créé)

### Le ReIDWorker asynchrone

DINOv2 met ~15-40 ms par batch. Pour ne pas bloquer la boucle vidéo, le Re-ID tourne dans un **thread séparé** qui consomme une file d'attente. La boucle vidéo pousse les crops et récupère les résultats de manière non-bloquante.

**Fichiers** : `reid.py`, `reid_worker.py`, `database.py`

---

## 5. Classification de race

### Le défi

Identifier la race d'un bovin sans entraîner de modèle spécifique. Aucun dataset annoté de races françaises n'existe publiquement.

### La solution : SigLIP-2 zero-shot

![Classification de race](diagram_breed.png)

**SigLIP-2 So400m** (Google, 1136M paramètres) projette images et texte dans un même espace vectoriel. Les races sont définies comme des **phrases descriptives** — aucune image de référence nécessaire.

### Fonctionnement

1. Au démarrage : embedding texte de chaque race (10 prompts)
2. Pour chaque nouveau bovin : embedding image du crop
3. Softmax sur les cosine similarities → probabilité par race

### Honnêteté : seuil de confiance

Si la race top-1 ne domine pas clairement (marge < 0.15), on affiche **« Indéterminée »** plutôt qu'une race potentiellement fausse.

**Fichier** : `breed.py`

---

## 6. Carte de densité (Heatmap)

La heatmap montre **où les bovins passent leur temps** dans le champ de la caméra.

![Principe de la heatmap](diagram_heatmap.png)

### Interprétation

| Couleur | Signification |
|---------|---------------|
| Rouge | Zone d'attroupement (mangeoire, point d'eau) |
| Jaune | Zone de passage |
| Noir | Zone jamais occupée |

### Implémentation

À chaque détection, la position normalisée (cx, cy) est enregistrée. L'analyseur agrège les ~2000 dernières positions dans une grille 20×20, affichée sur un canvas HTML5.

**Fichiers** : `analytics.py`, `app.js`

---

## 7. Interface utilisateur

L'interface est en HTML/CSS/JS pur (sans framework), servie directement par le worker Python.

```
web/public/
├── index.html      (structure)
├── styles.css      (thèmes dark/light)
├── app.js          (polling API, charts, heatmap)
└── splash.html     (écran de chargement)
```

### Composants

- **Flux vidéo annoté** : bounding boxes + noms + races
- **Sidebar** : liste des bovins actifs avec race et confiance
- **Dashboard Analytics** : KPIs, graphique FPS, répartition races, heatmap, timeline
- **Statistiques par bovin** : profil individuel (vidéos, activité, race)

---

## 8. Application desktop

### Pourquoi

Pour un usage professionnel : double-clic sur une icône, fenêtre native, démarrage/arrêt propre.

### Fonctionnement

1. Tauri affiche le splash screen
2. Lance le worker Python (`.venv/bin/python app.py`)
3. Attend le health check (TCP 127.0.0.1:8100)
4. Navigue vers l'interface
5. À la fermeture : tue le process Python proprement

### Cross-platform

| Plateforme | Format |
|-----------|--------|
| macOS | `.app` + `.dmg` |
| Windows | `.exe` (NSIS) + `.msi` |
| Linux | `.AppImage` + `.deb` |

**Fichiers** : `src-tauri/src/main.rs`, `src-tauri/tauri.conf.json`

---

## 9. Performance et optimisation

![Répartition du temps par frame](diagram_perf.png)

### Optimisations clés

| Optimisation | Gain |
|-------------|------|
| MLX (Metal GPU) | 3× plus rapide que CPU |
| ReIDWorker async | Re-ID ne bloque plus la boucle |
| Annotation ROI | -27 ms/frame |
| DINOv2 FP16 | 2× plus rapide |
| Cache matrice embeddings | match() vectorisé (BLAS) |

---

## 10. Installation et déploiement

### Prérequis

| Logiciel | Version | Rôle |
|----------|---------|------|
| Python | 3.10+ | Worker IA |
| Rust + Cargo | 1.97+ | Build Tauri |
| Xcode CLT (Mac) | — | Compilation native |

### Installation rapide

```bash
# Environnement Python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Lancer le worker
python app.py --mlx
# → http://localhost:8100

# App desktop
cargo install tauri-cli --version "^2.0"
cargo tauri dev
```

### Arguments principaux

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--source` | auto | Fichier vidéo ou webcam |
| `--mlx` | auto | Accélération Metal (Mac) |
| `--port` | 8100 | Port serveur |
| `--conf` | 0.4 | Seuil confiance YOLO |
| `--threshold` | 0.70 | Seuil similarité Re-ID |
| `--imgsz` | 640 | Taille inférence YOLO |

---

## 11. Fichiers du projet

```
boeuf-tracker/
├── app.py               Worker Flask (API + UI)
├── processor.py         Boucle de détection (cœur du système)
├── detector.py          Détection YOLO (MLX + PyTorch)
├── reid.py              Re-ID DINOv2
├── reid_worker.py       Thread asynchrone Re-ID
├── breed.py             Classification race (SigLIP-2)
├── database.py          Base d'embeddings (pickle)
├── names.py             Générateur de noms + compteur global
├── analytics.py         Stats (FPS, races, heatmap)
├── state.py             État global partagé
├── capture.py           Gestion sources vidéo
├── web/public/          Interface web (HTML/CSS/JS)
├── src-tauri/           Application desktop (Rust/Tauri)
├── requirements.txt     Dépendances Python
└── build.sh             Script de build desktop
```

---

## 12. Glossaire

| Terme | Définition |
|-------|-----------|
| YOLO | Réseau de détection d'objets temps réel |
| DINOv2 | Modèle self-supervised produisant des embeddings visuels |
| SigLIP-2 | Modèle vision-langage pour classification zero-shot |
| Embedding | Vecteur numérique représentant une image |
| Cosine similarity | Similarité entre vecteurs (0=différent, 1=identique) |
| Re-ID | Re-identification d'un individu à travers les frames |
| IoU | Intersection over Union (chevauchement de boîtes) |
| MLX | Framework ML Apple pour GPU Metal |
| Zero-shot | Classification sans entraînement spécifique |
| Heatmap | Carte de densité spatiale |

---

*PFE Individuel — Documentation Technique — Hiver 2026*
