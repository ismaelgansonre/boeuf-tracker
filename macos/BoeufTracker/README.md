# BoeufTracker — App macOS Swift ANE-first (Phase 1)

Application native macOS qui détecte, segmente et **compte les bœufs** en
exécutant YOLO11-seg sur le **Neural Engine (ANE)** via CoreML + Vision.

Source vidéo : **fichier** ou **caméra live**.

## Roadmap

| Phase | Contenu | État |
|-------|---------|------|
| **1** | Détection + comptage (YOLO-seg ANE), fichier + caméra | ✅ ce squelette |
| 2 | Tracking Swift + ReID (MegaDescriptor ANE) → identité | à venir |
| 3 | Posture (port de `posture.py` en Swift) | à venir |
| 4 | Comportement (X3D vidéo fine-tuné → CoreML) | à venir |

## 1. Générer le modèle CoreML

```bash
cd BoeufTracker/Conversion
pip install ultralytics coremltools
python export_yolo.py            # → yolo11s-seg.mlpackage
```

Renomme le `.mlpackage` en `YOLO11Seg.mlpackage` (ou adapte `modelName`
dans `Detector.swift`).

## 2. Créer le projet Xcode

Ce dossier contient le **code source**, pas encore un `.xcodeproj`
(il se génère depuis Xcode). Étapes :

1. Xcode → **File > New > Project > macOS > App**
   - Product Name : `BoeufTracker`
   - Interface : **SwiftUI**, Language : **Swift**
2. Supprime les fichiers générés par défaut (`ContentView.swift`, l'App).
3. Glisse les dossiers `Core/`, `UI/`, `BoeufTrackerApp.swift` dans le projet
   (coche *Copy items if needed*).
4. Glisse `YOLO11Seg.mlpackage` dans le projet (Xcode le compile en `.mlmodelc`).
5. **Signing & Capabilities** :
   - Coche **Camera** (accès caméra).
   - Info.plist → ajoute `NSCameraUsageDescription` = "Analyse du troupeau".
   - Pour ouvrir des fichiers hors sandbox : App Sandbox → **User Selected File : Read**.
6. **Run** (Cmd+R).

## 3. Vérifier que l'ANE est utilisé

Ouvre `YOLO11Seg.mlpackage` dans Xcode → onglet **Performance** →
*Generate Performance Report* → regarde la colonne **Compute Unit** :
les couches doivent afficher **Neural Engine**. Ce qui reste en GPU/CPU
= piste d'optimisation (souvent le pré/post-traitement).

## Architecture

```
FrameSource (fichier | caméra)  →  CVPixelBuffer
        │
        ▼
Detector (Vision + YOLO-seg, ANE)  →  [Detection]
        │
        ▼
Pipeline (@Observable)  →  count / detections / fps
        │
        ▼
LiveView (SwiftUI)  →  overlay boîtes + compteur
```

## Fichiers

- `Core/FrameSource.swift` — lecture fichier (AVAssetReader) + caméra (AVCaptureSession)
- `Core/Detector.swift` — YOLO-seg CoreML sur l'ANE via Vision
- `Core/Pipeline.swift` — orchestration + état observable, drop-frame temps réel
- `UI/LiveView.swift` — UI, sélection source, overlay détections + compteur
- `Conversion/export_yolo.py` — export YOLO11-seg → CoreML float16
