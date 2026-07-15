"""
processor.py
------------
Boucle principale de détection : lit la source, détecte + track + identifie
les bovins, annote la frame, met à jour STATE.

Auto-recovery en cas de crash (la boucle redémarre automatiquement).

Optimisations:
- Numba JIT sur le calcul de comportement (boucle Python serrée).
- Batcher l'embedding DINOv2 via reid.get_embedding_batch() pour tous les
  nouveaux tracks d'une frame en UN seul forward.
"""
import os
import time
import threading
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from numba import njit as _njit
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False

    def _njit(*args, **kwargs):  # type: ignore
        def deco(fn): return fn
        return deco

    # Import différé pour éviter une dépendance circulaire au moment du
    # fallback (console ne dépend de personne ici). On prévient explicitement
    # que les JIT sont désactivées pour que la dégradation de perf soit visible.
    try:
        from console import warn as _warn_fallback
        _warn_fallback(
            "[Numba] numba non installé — _total_displacement et "
            "_classify_behavior tournent en Python pur (×5-10 plus lent). "
            "Installez avec: pip install numba"
        )
    except Exception:
        pass


@_njit(cache=True, fastmath=True)
def _total_displacement(pts_x, pts_y):
    """Somme des distances euclidiennes entre points consécutifs. Numba JIT."""
    s = 0.0
    for i in range(1, pts_x.shape[0]):
        dx = pts_x[i] - pts_x[i - 1]
        dy = pts_y[i] - pts_y[i - 1]
        s += (dx * dx + dy * dy) ** 0.5
    return s


@_njit(cache=True, fastmath=True)
def _classify_behavior(speed: float, aspect: float, rel_y: float,
                       immobile_dur: float) -> int:
    """
    Retourne un code d'action (0..6). Numba JIT, ultra-rapide.
    0=couché, 1=pâture, 2=boit, 3=immobile, 4=marche, 5=court, 6=rué.
    """
    if aspect > 1.7 and speed < 5.0 and immobile_dur > 3.0:
        return 0
    if speed < 6.0 and aspect > 1.4:
        return 1
    if speed < 4.0 and aspect > 1.3 and rel_y > 0.6:
        return 2
    if speed < 5.0:
        return 3
    if speed < 25.0:
        return 4
    if speed < 80.0:
        return 5
    return 6


_BEHAVIOR_LABELS = ("couché", "pâture", "boit", "immobile", "marche", "court", "rué")


from console import info, ok, warn, err, dbg, evt, loop as log_loop
from detector import CattleDetector, CattleDetectorMLX
from reid import CattleReID
from database import EmbeddingDatabase
from reid_worker import ReIDWorker
from breed import classify_breed, get_clip_engine
from names import next_bovin_key, get_counter
from names import make_name_generator
from analytics import init as init_analytics, get as get_analytics, DetectionSample
from state import STATE, color_for_name, reset_for_new_source
from capture import open_capture, source_label


def annotate_frame(annotated, masks_data, det_idx, x1, y1, x2, y2, color):
    """Dessine le masque de segmentation (silhouette) si dispo, sinon rectangle.

    Optimisé : on ne traite que la région de la bounding box (ROI) au lieu de
    toute l'image. Évite d'allouer un np.zeros_like(full_frame) par bovin et
    un addWeighted sur l'image complète — ces deux ops coûtaient ~29ms/frame.
    """
    mask_drawn = False
    if masks_data is not None and det_idx < len(masks_data):
        try:
            m = masks_data[det_idx]
            H, W = annotated.shape[:2]
            # Resize du masque une seule fois à la taille de l'image
            mask_resized = cv2.resize(
                m, (W, H), interpolation=cv2.INTER_LINEAR,
            )
            bin_mask = (mask_resized > 0.5).astype(np.uint8)
            # ROI = bounding box + petite marge : on ne travaille que dessus
            rx1, ry1 = max(0, x1 - 4), max(0, y1 - 4)
            rx2, ry2 = min(W, x2 + 4), min(H, y2 + 4)
            roi = annotated[ry1:ry2, rx1:rx2]
            roi_mask = bin_mask[ry1:ry2, rx1:rx2]
            # Tint subtil (8%) uniquement sur la ROI
            tint = np.zeros_like(roi)
            tint[roi_mask == 1] = color
            annotated[ry1:ry2, rx1:rx2] = cv2.addWeighted(roi, 1.0, tint, 0.08, 0)
            # Contour net sur la ROI (coords relatives)
            contours, _ = cv2.findContours(
                roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(annotated[ry1:ry2, rx1:rx2], contours, -1, color, 2)
            mask_drawn = True
        except Exception:
            pass
    if not mask_drawn:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    return annotated


def analyze_behavior(boxes, track_ids, t_now, frame_shape=None):
    """Catégorise l'activité: pâture, boit, couché, immobile, marche, court, rué.

    Optimisé: les calculs lourds (somme de distances, classification) sont JIT
    Numba → ~5-10× plus rapide que le pure Python sur la boucle interne.
    """
    behaviors = []
    frame_h = frame_shape[0] if frame_shape is not None else 1080
    for box, tid in zip(boxes, track_ids):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        aspect = bw / max(bh, 1)
        rel_y = cy / frame_h  # 0 = haut, 1 = bas

        hist = STATE["track_history"].setdefault(int(tid), [])
        hist.append((cx, cy, t_now))
        STATE["track_history"][int(tid)] = [p for p in hist if t_now - p[2] < 5.0]
        pts = STATE["track_history"][int(tid)]
        if len(pts) < 3:
            continue

        # Vitesse instantanée (sur la fenêtre historique)
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        dist = (dx * dx + dy * dy) ** 0.5
        dt = max(pts[-1][2] - pts[0][2], 1e-3)
        speed = dist / dt

        # Displacement cumulé (JIT) + classification (JIT)
        pts_arr = np.asarray(pts, dtype=np.float32)
        total_disp = _total_displacement(pts_arr[:, 0], pts_arr[:, 1])
        immobile_since = pts[0][2] if total_disp < 30 else None
        immobile_dur = (t_now - immobile_since) if immobile_since is not None else 0.0

        action_code = _classify_behavior(float(speed), float(aspect),
                                         float(rel_y), float(immobile_dur))
        action = _BEHAVIOR_LABELS[action_code]

        behaviors.append({
            "name": STATE.get("_track_names", {}).get(int(tid), "?"),
            "action": action,
            "speed": round(float(speed), 1),
            "track_id": int(tid),
            "aspect": round(float(aspect), 2),
        })
    return behaviors


def _mlx_available() -> bool:
    """Vérifie si MLX (Apple Metal GPU) est disponible.

    Source unique de vérité (utilisée aussi par app.py via import) — ne pas
    dupliquer ailleurs. MLX >= 0.30 exige un argument device ; les versions
    plus anciennes ne prenaient aucun argument. On gère les deux pour rester
    robuste aux futures montées de version.
    """
    try:
        import mlx.core as mx
        try:
            return bool(mx.is_available(mx.gpu))  # MLX 0.30+
        except TypeError:
            return bool(mx.is_available())  # vieille API MLX (< 0.30)
    except Exception:
        return False


def resolve_device(requested: str, for_pytorch: bool = False) -> str:
    """Résout le device de calcul.

    for_pytorch: si True, on ne retourne JAMAIS 'mlx' car PyTorch (DINOv2)
    ne supporte que cpu/cuda/mps. MLX est un backend séparé réservé à YOLO26.
    Utilisé pour le device du Re-ID (DINOv2) quand le détecteur tourne sur MLX.
    """
    # MLX = priorité maximale sur Apple Silicon (YOLO uniquement)
    if requested in ("auto", "mlx") and not for_pytorch:
        if _mlx_available():
            return "mlx"
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    # Device PyTorch (DINOv2) : on saute mlx, on prend cuda > mps > cpu
    if requested in ("auto", "mlx") and for_pytorch:
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu"
        if requested == "cuda":
            return "cuda:0"
        try:
            idx = int(requested.split(":")[1])
            if 0 <= idx < torch.cuda.device_count():
                return f"cuda:{idx}"
        except (ValueError, IndexError):
            pass
        return "cuda:0"
    return "cpu"


def detection_loop(args):
    """Boucle principale avec auto-recovery."""
    source = int(args.source) if args.source.isdigit() else args.source
    STATE["source"] = str(source)
    STATE["current_source_path"] = source
    STATE["source_label"] = source_label(source)
    STATE["started_at"] = datetime.now().isoformat(timespec="seconds")

    # MLX: YOLO26 sur Metal GPU Apple (2.6× plus rapide que PyTorch MPS)
    # Auto-détection : si MLX est dispo et qu'on est en mode "auto", on
    # l'utilise automatiquement (backend natif le plus rapide). On bascule
    # aussi le modèle par défaut vers le modèle MLX si nécessaire.
    use_mlx = getattr(args, "mlx", False)
    if not use_mlx and args.device in ("auto", "mlx") and _mlx_available():
        # Vérifier qu'un modèle MLX (.safetensors ou yolo26) est disponible
        model_is_mlx = (
            args.yolo_model.endswith(".safetensors")
            or args.yolo_model.startswith("yolo26")
        )
        if model_is_mlx:
            use_mlx = True
            ok(f"[Init] MLX auto-detecté (modèle {args.yolo_model} compatible). "
               f"Activation du backend Metal natif.")
        else:
            # Modèle PyTorch (yolo11*.pt) mais MLX dispo : on garde PyTorch
            # sur MPS. resolve_device doit retourner mps, pas mlx.
            ok(f"[Init] MLX dispo mais modèle {args.yolo_model} est PyTorch. "
               f"Utilisation de MPS.")

    if use_mlx:
        device = "mlx"
        STATE["device"] = "mlx"
        ok(f"[Init] Device: mlx (YOLO26 Metal GPU)")
        # DINOv2 (Re-ID) est PyTorch : il ne supporte PAS "mlx", seulement
        # mps/cpu/cuda. resolve_device(for_pytorch=True) exclut mlx.
        if args.device != "auto":
            reid_device = resolve_device(args.device, for_pytorch=True)
        else:
            reid_device = resolve_device("auto", for_pytorch=True)
    else:
        device = resolve_device(args.device)
        # Si le device résolu est "mlx" mais qu'on est ici (modèle PyTorch),
        # on ne peut pas l'utiliser — on bascule sur le meilleur device PyTorch.
        if device == "mlx":
            device = resolve_device("auto", for_pytorch=True)
        STATE["device"] = device
        ok(f"[Init] Device: {device}")
        if device.startswith("cuda"):
            try:
                idx = int(device.split(":")[1]) if ":" in device else 0
                ok(f"[Init] GPU: {torch.cuda.get_device_name(idx)}")
            except Exception:
                pass
        reid_device = device

    # Modèles
    if use_mlx:
        info(f"[YOLO26-MLX] Utilisation de CattleDetectorMLX...")
        # When --mlx, use yolo26s-seg.safetensors (or the user-specified model)
        mlx_model = args.yolo_model if args.yolo_model != "yolo11s-seg.pt" else "yolo26s-seg.safetensors"
        detector = CattleDetectorMLX(model_name=mlx_model, device="mlx")
    else:
        detector = CattleDetector(model_name=args.yolo_model, device=device)
    reid = CattleReID(model_name=args.dino_model, device=reid_device)
    db = EmbeddingDatabase(path=args.db, reid_engine=reid)
    # Synchronise le compteur global avec la DB existante : si la DB contient
    # Boeuf_012 mais le compteur est a 0, on remonte le compteur a 12 pour
    # garantir que le prochain bovin sera Boeuf_013 (jamais de doublon).
    counter = get_counter()
    max_key_num = 0
    for key in db.animals:
        if key.startswith("Boeuf_"):
            try:
                max_key_num = max(max_key_num, int(key.split("_")[1]))
            except (ValueError, IndexError):
                pass
    if counter.peek() < max_key_num:
        counter.value = max_key_num
        counter._save()
        ok(f"[Names] Compteur global synchronisé sur la DB: {max_key_num} bovins connus")
    # Worker asynchrone : découple DINOv2 de la boucle vidéo pour garantir
    # un FPS stable (le Re-ID ne bloque plus la détection).
    reid_worker = ReIDWorker(reid)
    ok(f"[ReIDWorker] Thread asynchrone démarré (decouplage DINOv2)")
    # Générateur de noms propres (stable cross-session via first_seen)
    name_gen = make_name_generator(db)
    ok(f"[Names] {len(name_gen.all())} noms attribues à partir de la DB")
    STATE["name_gen"] = name_gen
    # Analytics : accumule FPS, races, positions pour le dashboard
    analytics = init_analytics(lambda: STATE)
    ok(f"[Analytics] Collecteur démarré (sampling 2s, dashboard + heatmap)")
    # Pre-charge CLIP (classification de race zero-shot) pour eviter un delai
    # de ~15s au moment du 1er nouvel animal detecte. Non bloquant : si CLIP
    # est indisponible, on retombe sur le fallback HSV de breed.classify_breed.
    try:
        engine = get_clip_engine()
        if engine is not None:
            ok(f"[Breed] CLIP zero-shot pret ({len(engine.race_names)} races)")
        else:
            warn("[Breed] CLIP indisponible, fallback HSV active")
    except Exception as e:
        warn(f"[Breed] Pre-charge CLIP echoue ({e}), fallback HSV")

    # Validation compatibilité dim
    dummy_crop = np.zeros((128, 128, 3), dtype=np.uint8)
    sample_emb = reid.get_embedding(dummy_crop)
    expected_dim = int(sample_emb.shape[0]) if sample_emb is not None else reid.TOTAL_DIM
    remaining = db.validate_dim(expected_dim)
    ok(f"[DB] {remaining} animaux charges (dim={expected_dim})")

    # Sync initial settings to STATE (sinon les boutons UI ne savent pas l'état réel)
    STATE["yolo_model_current"] = args.yolo_model
    STATE["imgsz_current"] = args.imgsz
    STATE["embed_every_current"] = args.embed_every
    STATE["threshold_current"] = args.threshold
    STATE["conf_current"] = args.conf

    # Capture initiale
    cap = open_capture(source)
    if cap is None:
        err(f"[Init] Source introuvable: {source}")
        return

    track_id_to_name = {}
    STATE["_track_names"] = track_id_to_name  # pour behavior
    track_emb_accum = {}
    fps_smooth = 0.0
    frame_idx = 0

    # Anti-double-comptage sur vidéo en boucle
    prev_cap_pos: int = -1
    loop_detected_at_frame: int = -10**9  # frame_idx du dernier rebobinage
    max_updates_per_animal: int = max(1, int(getattr(args, "max_updates", 30)))
    loop_threshold: float = float(getattr(args, "loop_threshold", 0.45))
    loop_grace_frames: int = max(1, int(getattr(args, "loop_grace_frames", 60)))

    def switch_device(new_dev):
        try:
            info(f"[Device] Switch {STATE['device']} -> {new_dev}")
            detector.device = new_dev
            reid.model = reid.model.to(new_dev)
            reid.device = new_dev
            with torch.no_grad():
                dummy_pil = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
                dummy_in = reid.processor(images=dummy_pil, return_tensors="pt").to(new_dev)
                _ = reid.model(**dummy_in)
            STATE["device"] = new_dev
            STATE["events"].insert(0, f"DEVICE -> {new_dev}")
            STATE["events"] = STATE["events"][:30]
            ok(f"[Device] OK sur {new_dev}")
            return True
        except Exception as e:
            err(f"[Device] ERREUR: {e}")
            return False

    def apply_desired_settings():
        """Applique les changements demandes via /api/settings a chaud."""
        nonlocal detector

        # imgsz (changement instantane, pas de reload)
        d = STATE.get("desired_imgsz")
        if d is not None and d != STATE.get("imgsz_current"):
            args.imgsz = d
            STATE["imgsz_current"] = d
            STATE["desired_imgsz"] = None
            STATE["events"].insert(0, f"IMGSZ -> {d}")
            STATE["events"] = STATE["events"][:30]
            ok(f"[Settings] imgsz -> {d}")

        # embed_every (pas de reload, juste frequence)
        d = STATE.get("desired_embed_every")
        if d is not None and d != STATE.get("embed_every_current"):
            STATE["embed_every_current"] = d
            STATE["desired_embed_every"] = None
            STATE["events"].insert(0, f"EMBED_EVERY -> {d}")
            STATE["events"] = STATE["events"][:30]
            ok(f"[Settings] embed_every -> {d}")

        # threshold (pas de reload)
        d = STATE.get("desired_threshold")
        if d is not None and abs(d - STATE.get("threshold_current", 0)) > 1e-6:
            args.threshold = d
            STATE["threshold_current"] = d
            STATE["desired_threshold"] = None
            STATE["events"].insert(0, f"THRESHOLD -> {d:.2f}")
            STATE["events"] = STATE["events"][:30]
            ok(f"[Settings] threshold -> {d:.2f}")

        # conf (pas de reload)
        d = STATE.get("desired_conf")
        if d is not None and abs(d - STATE.get("conf_current", 0)) > 1e-6:
            args.conf = d
            STATE["conf_current"] = d
            STATE["desired_conf"] = None
            STATE["events"].insert(0, f"CONF -> {d:.2f}")
            STATE["events"] = STATE["events"][:30]
            ok(f"[Settings] conf -> {d:.2f}")

        # yolo_model (reload du modele, ~2-5s de freeze)
        d = STATE.get("desired_yolo_model")
        if d is not None and d != STATE.get("yolo_model_current"):
            info(f"[YOLO] Reload {STATE.get('yolo_model_current')} -> {d}")
            STATE["events"].insert(0, f"YOLO reload -> {d}")
            STATE["events"] = STATE["events"][:30]
            try:
                old = detector
                # Choisit le bon detecteur selon le type de fichier
                is_mlx_model = d.endswith(".safetensors") or d.startswith("yolo26")
                if is_mlx_model:
                    new_det = CattleDetectorMLX(model_name=d, device="mlx")
                else:
                    # Convertit le device MLX en device PyTorch compatible (mps/cpu)
                    pt_device = resolve_device("auto", for_pytorch=True)
                    new_det = CattleDetector(model_name=d, device=pt_device, half=False)
                detector = new_det
                STATE["yolo_model_current"] = d
                STATE["desired_yolo_model"] = None
                STATE["events"].insert(0, f"YOLO OK: {d}")
                STATE["events"] = STATE["events"][:30]
                ok(f"[YOLO] Reload OK -> {d}")
            except Exception as e:
                STATE["desired_yolo_model"] = None  # clear to avoid loop
                STATE["events"].insert(0, f"YOLO ERREUR: {e}")
                STATE["events"] = STATE["events"][:30]
                err(f"[YOLO] Reload ERREUR: {e}")

    # Boucle externe: récupère les crashes
    while True:
        try:
            while True:
                # Applique les changements live (imgsz, embed_every, threshold, conf, model)
                apply_desired_settings()

                # Device switch
                desired_dev = STATE.get("desired_device")
                if desired_dev is not None and desired_dev != STATE.get("device"):
                    switch_device(desired_dev)
                    STATE["desired_device"] = None

                # Source switch
                desired = STATE.get("desired_source")
                if desired is not None:
                    new_src = int(desired) if (isinstance(desired, str) and desired.isdigit()) else desired
                    info(f"[Switch] -> {new_src}")
                    cap.release()
                    new_cap = open_capture(new_src)
                    if new_cap is not None:
                        cap = new_cap
                        STATE["current_source_path"] = new_src
                        STATE["source"] = str(new_src)
                        STATE["source_label"] = source_label(new_src)
                        track_id_to_name.clear()
                        track_emb_accum.clear()
                        STATE["_track_names"] = track_id_to_name
                        reset_for_new_source()
                        ok(f"[Switch] OK: {STATE['source_label']}")
                    else:
                        cap = open_capture(STATE["current_source_path"]) or cap
                        err("[Switch] ERREUR ouverture")
                    STATE["desired_source"] = None

                # Re-appariement demande via /api/rematch (force une re-id de tous
                # les tracks en cours avec le seuil courant).
                if STATE.pop("desired_rematch", False):
                    info("[Rematch] Vidage du cache tracks -> re-id au prochain passage")
                    STATE["events"].insert(0, "REMATCH -> prochaine frame")
                    STATE["events"] = STATE["events"][:30]
                    track_id_to_name.clear()
                    track_emb_accum.clear()
                    STATE["_track_names"] = track_id_to_name

                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    src = STATE["current_source_path"]
                    if isinstance(src, int):
                        # Webcam: reconnect
                        try:
                            cap.release()
                        except Exception:
                            pass
                        time.sleep(2.0)
                        new_cap = open_capture(src)
                        if new_cap is None:
                            time.sleep(3.0)
                            new_cap = open_capture(src)
                        cap = new_cap or cap
                    else:
                        # Fichier: rebobiner
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # Sur webcam on n'invalide pas prev_cap_pos (POS_FRAMES est -1).
                    # Sur fichier rebobine, le check <prev_cap_pos ci-dessous le detectera.
                    continue  # on re-tentera a la prochaine iteration

                # Detection d'un rebobinage de fichier (frame_pos qui regresse).
                # On reset le tracker pour forcer une re-id propre par embedding
                # et on evite ainsi le double-comptage d'un meme bovin.
                if not isinstance(STATE["current_source_path"], int):
                    try:
                        cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    except Exception:
                        cur_pos = -1
                    if prev_cap_pos > 0 and 0 <= cur_pos < prev_cap_pos:
                        log_loop(
                            f"Rebobinage detecte ({prev_cap_pos} -> {cur_pos}), "
                            f"reset du tracker"
                        )
                        STATE["events"].insert(0, "LOOP -> tracker reset")
                        STATE["events"] = STATE["events"][:30]
                        track_id_to_name.clear()
                        track_emb_accum.clear()
                        STATE["_track_names"] = track_id_to_name
                        loop_detected_at_frame = frame_idx
                    if cur_pos > 0:
                        prev_cap_pos = cur_pos

                # Skip frames pour économie GPU
                if args.skip_frames > 0 and frame_idx % (args.skip_frames + 1) != 0:
                    STATE["frame_count"] = frame_idx
                    frame_idx += 1
                    continue

                # Détection + tracking + segmentation
                result = detector.detect(frame, persist=True, conf=args.conf, imgsz=args.imgsz)
                annotated = frame.copy()
                active = []

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    track_ids = result.boxes.id.int().cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    masks_data = (
                        result.masks.data.cpu().numpy()
                        if result.masks is not None and len(result.masks) > 0
                        else None
                    )

                    # Noms déjà attribués dans cette frame (pour exclusion)
                    frame_names = {
                        track_id_to_name[t]
                        for t in track_ids
                        if int(t) in track_id_to_name and track_id_to_name[int(t)] != "?"
                    }

                    # Phase 1: collecte des crops éligibles à l'embedding
                    # (1 seul forward DINOv2 batché au lieu de N forwards)
                    reid_every = max(1, int(STATE.get("embed_every_current", 30)))
                    new_track_indices: list[int] = []     # nouveaux tracks
                    reembed_track_indices: list[int] = []  # tracks existants (EMA)
                    # Clé STABLE = track_id (int). det_idx change à chaque frame
                    # et ne peut pas servir de clé pour le worker asynchrone.
                    crops_to_submit: dict[int, np.ndarray] = {}  # {track_id: crop}
                    det_idx_to_tid: dict[int, int] = {}

                    for det_idx, (box, tid, conf) in enumerate(zip(boxes, track_ids, confs)):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if (x2 - x1) < 24 or (y2 - y1) < 24:
                            continue
                        det_idx_to_tid[det_idx] = int(tid)
                        crop = frame[y1:y2, x1:x2]
                        # "?" = en attente d'embedding (worker pas encore prêt).
                        # On le traite comme un nouveau track pour le re-soumettre.
                        existing = track_id_to_name.get(int(tid))
                        if existing is None or existing == "?":
                            new_track_indices.append(det_idx)
                            crops_to_submit[int(tid)] = crop
                        elif frame_idx % reid_every == 0:
                            reembed_track_indices.append(det_idx)
                            crops_to_submit[int(tid)] = crop

                    # Phase 2: soumission asynchrone + collecte des embeddings prêts.
                    # Le worker calcule DINOv2 dans un thread séparé : la boucle
                    # vidéo n'attend jamais le forward. Les crops non prêts cette
                    # frame seront traités à la suivante.
                    # emb_by_tid: {track_id: embedding} — clé stable cross-frames
                    if reid_worker.has_failed():
                        # Fallback synchrone si le worker a crashé
                        if crops_to_submit:
                            embeddings = reid.get_embedding_batch(
                                list(crops_to_submit.values())
                            )
                        else:
                            embeddings = []
                        emb_by_tid: dict[int, np.ndarray] = {}
                        for k, (tid, _) in enumerate(crops_to_submit.items()):
                            e = embeddings[k] if k < len(embeddings) else None
                            if e is not None and e.shape[0] == expected_dim:
                                emb_by_tid[tid] = e
                    else:
                        # Soumet les crops au worker, clé = track_id (non-bloquant)
                        reid_worker.submit_batch(crops_to_submit)
                        # Récupère ce qui est prêt (non-bloquant, peut être vide)
                        emb_by_tid = reid_worker.collect_ready()

                    # Phase 3: matching / EMA / annotation
                    for det_idx, (box, tid, conf) in enumerate(zip(boxes, track_ids, confs)):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if (x2 - x1) < 24 or (y2 - y1) < 24:
                            continue
                        tid_int = int(tid)

                        if det_idx in new_track_indices:
                            emb = emb_by_tid.get(tid_int)
                            if emb is not None:
                                in_loop_grace = (
                                    frame_idx - loop_detected_at_frame
                                ) < loop_grace_frames
                                eff_threshold = (
                                    loop_threshold if in_loop_grace else args.threshold
                                )
                                name, sim = db.match(
                                    emb,
                                    threshold=eff_threshold,
                                    exclude=frame_names,
                                )
                                event = None
                                if name is None:
                                    # Cle unique via compteur global persistant :
                                    # garantit que les noms ne sont JAMAIS
                                    # reutilises, meme apres un reset de DB.
                                    name = next_bovin_key()
                                    # Identification du robe-type par analyse HSV
                                    # (breed.py → coat_type + breeds compatibles).
                                    # Instantané (<0.5ms), réutilise le crop courant.
                                    crop = frame[y1:y2, x1:x2]
                                    breed_result = classify_breed(crop)
                                    coat_type = breed_result.get("coat_type", "Indeterminee")
                                    breed_conf = breed_result.get("confidence", 0.0)
                                    db.add(name, emb,
                                           breed=coat_type,
                                           breed_confidence=breed_conf,
                                           coat_swatch=breed_result.get("swatch"),
                                           breeds_compat=[b["name"] for b in breed_result.get("breeds", [])])
                                    # Met a jour le generateur de noms avec le nouveau bovin
                                    name_gen.__init__(db.animals)
                                    STATE["name_gen"] = name_gen
                                    # Persiste la DB sur disque a chaque nouvel animal
                                    # (pickle, ~1-2ms, garantit la stabilite cross-session)
                                    try:
                                        db.save()
                                    except Exception as e:
                                        warn(f"[DB] save failed: {e}")
                                    proper_name = name_gen.get(name)
                                    event = (
                                        f"NEW  {proper_name} ({name})  {coat_type} "
                                        f"(sim_max={sim:.3f}, robe={breed_conf:.0%})"
                                    )
                                else:
                                    proper_name = name_gen.get(name)
                                    event = (
                                        f"MATCH {proper_name} ({name})  "
                                        f"(sim={sim:.3f}, thr={eff_threshold:.2f})"
                                    )
                                track_id_to_name[int(tid)] = name
                                track_emb_accum[int(tid)] = [emb]
                                STATE["_track_names"] = track_id_to_name
                                frame_names.add(name)
                                if event:
                                    STATE["events"].insert(0, event)
                                    STATE["events"] = STATE["events"][:30]
                            else:
                                track_id_to_name[int(tid)] = "?"
                                STATE["_track_names"] = track_id_to_name
                        elif det_idx in reembed_track_indices:
                            emb = emb_by_tid.get(tid_int)
                            if emb is not None:
                                buf = track_emb_accum.setdefault(int(tid), [])
                                buf.append(emb)
                                if len(buf) > 4:
                                    track_emb_accum[int(tid)] = buf[-4:]
                                    buf = track_emb_accum[int(tid)]
                                if len(buf) >= 2:
                                    stacked = np.stack(buf[-2:]).astype(np.float64)
                                    mean = stacked.mean(axis=0)
                                    mean = mean / (np.linalg.norm(mean) + 1e-8)
                                    aname = track_id_to_name[int(tid)]
                                    cur = (
                                        db.animals.get(aname, {}).get("count", 0)
                                        if isinstance(db.animals, dict) and aname in db.animals
                                        else 0
                                    )
                                    if cur < max_updates_per_animal:
                                        db.update(
                                            aname, mean.astype(np.float32)
                                        )

                        name = track_id_to_name.get(int(tid), "?")
                        color = color_for_name(name)
                        annotated = annotate_frame(
                            annotated, masks_data, det_idx,
                            x1, y1, x2, y2, color,
                        )

                        # Race stockée en DB (pour les animaux identifiés)
                        animal_data = db.animals.get(name, {}) if name != "?" else {}
                        breed_name = animal_data.get("breed") or "Indeterminee"
                        breed_conf = animal_data.get("breed_confidence", 0)

                        # Nom propre (stable cross-session) via NameGenerator
                        display_name = name_gen.get(name) if name != "?" else "?"

                        # Label avec nom propre + race (compact)
                        label = f"{display_name}  {breed_name}  {float(conf):.2f}"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        ly1 = max(0, y1 - th - 10)
                        ly2 = ly1 + th + 10
                        cv2.rectangle(annotated, (x1, ly1), (x1 + tw, ly2), color, -1)
                        cv2.putText(annotated, label, (x1, ly1 + th + 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        active.append({
                            "name": display_name,
                            "key": name,  # la cle interne (Boeuf_001), pour debug
                            "conf": float(conf),
                            "track_id": int(tid),
                            "breed": breed_name,
                            "breed_confidence": round(float(breed_conf), 2) if breed_conf else 0,
                        })

                        # Pousse la position (normalisee 0-1) vers le collecteur
                        # analytics pour la heatmap. Non-bloquant, decimation
                        # interne au collector si >5000 samples.
                        if name != "?":
                            fh, fw = frame.shape[:2]
                            cx_norm = (x1 + x2) / 2.0 / max(fw, 1)
                            cy_norm = (y1 + y2) / 2.0 / max(fh, 1)
                            # Behavior lookup
                            behavior_now = "active"
                            for b in STATE.get("behavior", []):
                                if b.get("track_id") == int(tid):
                                    behavior_now = b.get("action", "active")
                                    break
                            try:
                                analytics.push_detection(DetectionSample(
                                    frame=frame_idx,
                                    proper_name=display_name,
                                    key=name,
                                    breed=breed_name,
                                    conf=float(conf),
                                    cx_norm=cx_norm,
                                    cy_norm=cy_norm,
                                    behavior=behavior_now,
                                ))
                            except Exception:
                                pass

                    # Comportement (sur la dernière frame traitée)
                    STATE["behavior"] = analyze_behavior(boxes, track_ids, time.time(), frame.shape)

                # Encodage JPEG
                enc_ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if enc_ok:
                    with STATE["frame_lock"]:
                        STATE["frame_jpg"] = buf.tobytes()
                        STATE["active_animals"] = active

                elapsed = time.time() - t0
                fps_inst = 1.0 / max(elapsed, 1e-6)
                fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst if fps_smooth else fps_inst
                STATE["fps"] = fps_smooth
                STATE["frame_count"] = frame_idx
                frame_idx += 1

        except Exception as e:
            err(f"[Crash] {type(e).__name__}: {e}")
            import traceback
            tb = traceback.format_exc()
            print(tb, flush=True)
            # Garde le message d'erreur + dernière ligne du traceback pour debug
            last_tb = tb.strip().splitlines()[-1] if tb else str(e)
            STATE["events"].insert(0, f"CRASH: {type(e).__name__}: {str(e)[:80]} | {last_tb[:60]}")
            STATE["events"] = STATE["events"][:30]
            time.sleep(2)
            try:
                cap.release()
            except Exception:
                pass
            cap = open_capture(STATE["current_source_path"]) or cv2.VideoCapture(0)
            continue


def _recover_capture(cap, src, ret):
    """Helper legacy (gardé pour compat)."""
    if ret:
        return cap
    if isinstance(src, int):
        cap.release()
        time.sleep(2.0)
        new_cap = open_capture(src)
        if new_cap is None:
            time.sleep(3.0)
            new_cap = open_capture(src)
        return new_cap or cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap


def start_detection_thread(args) -> threading.Thread:
    """Démarre le thread de détection. À appeler une seule fois au boot."""
    t = threading.Thread(target=detection_loop, args=(args,), daemon=True)
    t.start()
    return t


# Imports internes pour switch_device