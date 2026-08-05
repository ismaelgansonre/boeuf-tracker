#!/usr/bin/env python3
"""
export_yolo.py
--------------
Exporte YOLO11-seg en CoreML float16, optimisé pour le Neural Engine (ANE).

ANE-first :
  - half=True        → poids float16 (l'ANE est natif fp16 ; fp32 tombe sur GPU/CPU).
  - imgsz=640        → entrée FIXE 640x640 (shapes dynamiques cassent l'ANE).
  - nms=True         → NMS intégré au modèle → Vision renvoie des
                       VNRecognizedObjectObservation directement exploitables.

Usage :
    pip install ultralytics coremltools
    python export_yolo.py                 # yolo11s-seg par défaut
    python export_yolo.py yolo11n-seg.pt  # variante plus légère

Sortie : yolo11s-seg.mlpackage  → à glisser dans le projet Xcode (Models/),
         puis renommer/référencer comme "YOLO11Seg" dans Detector.swift.

Vérifier l'ANE ensuite : ouvrir le .mlpackage dans Xcode →
onglet "Performance" → générer un rapport → colonne "Compute Unit".
"""
import sys
from ultralytics import YOLO


def main() -> None:
    weights = sys.argv[1] if len(sys.argv) > 1 else "yolo11s-seg.pt"
    print(f"[export] chargement de {weights}")
    model = YOLO(weights)

    print("[export] → CoreML float16, 640x640, NMS intégré")
    model.export(
        format="coreml",
        half=True,       # float16 pour l'ANE
        nms=True,        # NMS dans le graphe
        imgsz=640,       # entrée fixe
        # Ce YOLO reste sur les classes COCO ; 'cow' = classe 19.
        # Après fine-tuning sur tes bœufs, ré-exporter avec les mêmes flags.
    )
    print("[export] OK → *.mlpackage généré dans le dossier courant")


if __name__ == "__main__":
    main()
