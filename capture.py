"""
capture.py
----------
Helpers pour ouvrir une source vidéo avec le bon backend et basculer
entre webcam / fichier / flux RTSP à chaud.
"""
import time
import cv2


def open_capture(src):
    """
    Ouvre une source vidéo de manière robuste.
    - Sur Windows, essaie DirectShow avant MSMF (plus stable pour webcam)
    - Pour les fichiers, utilise le backend par défaut
    Retourne un VideoCapture ouvert ou None.
    """
    if isinstance(src, int):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            try:
                cap = cv2.VideoCapture(src, backend)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        return cap
                    cap.release()
            except Exception:
                pass
        return None
    cap = cv2.VideoCapture(src)
    return cap if cap.isOpened() else None


def read_with_recovery(cap, src):
    """
    Lit une frame avec auto-recovery si la source se coupe.
    - Pour webcam (int): retente après 2s, puis 3s, en boucle
    - Pour fichier: rebobine au début (mode boucle)
    Retourne (ret, frame).
    """
    ret, frame = cap.read()
    if ret:
        return ret, frame
    if isinstance(src, int):
        print("[Webcam] Flux perdu, reconnexion...", flush=True)
        cap.release()
        time.sleep(2.0)
        new_cap = open_capture(src)
        if new_cap is None:
            time.sleep(3.0)
            new_cap = open_capture(src)
        if new_cap is not None:
            return False, None  # signal que la capture a été remplacée
        return False, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return False, None


def source_label(src) -> str:
    """Génère un label lisible pour affichage."""
    if isinstance(src, int):
        return f"webcam {src}"
    if isinstance(src, str):
        name = src.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        # Enlever préfixe timestamp "1234567890_"
        if "_" in name:
            parts = name.split("_", 1)
            if parts[0].isdigit() and len(parts[0]) >= 10:
                name = parts[1]
        return name
    return str(src)