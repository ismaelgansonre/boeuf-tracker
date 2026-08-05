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

from console import info, ok, warn, err, dbg, evt, loop as log_loop
from behavior import (
    BehaviorAnalyzer, EventJournal, IsolationMonitor, TransitionMonitor,
    NUMBA_OK as _NUMBA_OK,
)
from posture import MaskHeadAnalyzer, HeadMotionTracker, overlap_fractions
from posture_model import PostureModel
from behavior_video import VideoBehavior

# Marge de similarite requise pour retirer un nom a la piste qui le porte.
# Trop bas -> les noms sautent d'une piste a l'autre ; trop haut -> le vrai
# bovin ne peut jamais reprendre son nom apres une occlusion.
REASSIGN_MARGIN = 0.05
# Au-dela de cette fraction de recouvrement par un congenere, l'imagette est
# consideree polluee : on ne s'en sert pas pour mettre a jour l'empreinte.
OCCLUSION_SKIP_FRAC = 0.25
# Distance au bord bas de l'image en deca de laquelle on considere la
# silhouette tronquee (pattes hors champ) — cf. posture.MaskHeadAnalyzer.
BOTTOM_EDGE_MARGIN_PX = 4
# Cote minimal d'une imagette exploitable par la re-identification, en pixels.
# 24 px excluait les bovins d'arriere-plan que la montee de `imgsz` fait
# justement apparaitre : ils etaient detectes mais jamais nommes ni annotes.
MIN_CROP_SIDE_PX = 16
# Age minimal (en frames vues) d'une piste avant de pouvoir CREER une nouvelle
# identite en base. Les pistes ephemeres (doublon de detection, artefact d'une
# poignee de frames) creaient chacune un bovin fantome : mesure sur
# samples/IMG_3544, 32 identites pour ~12 animaux reels. Une piste jeune reste
# "?" ; le rattachement a un bovin DEJA connu, lui, est immediat.
MIN_NEW_ID_AGE_FRAMES = 12
# Avant de creer une identite, on retente un appariement au seuil abaisse de
# cette marge, restreint aux bovins hors champ : mieux vaut rattacher une
# piste a un animal connu legerement different (angle, lumiere) que gonfler
# le comptage avec un doublon.
SECOND_CHANCE_DELTA = 0.10
if not _NUMBA_OK:  # pragma: no cover
    warn("[Numba] non installé — JIT désactivé, comportement ×5-10 plus lent. "
         "pip install numba")
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

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
    _SKELETON_OK = True
except Exception:
    _SKELETON_OK = False


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

            # Squelette morphologique (option UI). Reduit la silhouette a
            # une ligne mediane de 1 px, puis on la dessine par-dessus.
            if _SKELETON_OK and STATE.get("show_skeleton"):
                try:
                    sk = _sk_skeletonize(roi_mask.astype(bool)).astype(np.uint8)
                    ys, xs = np.where(sk > 0)
                    if len(xs) > 0:
                        roi_view = annotated[ry1:ry2, rx1:rx2]
                        roi_view[ys, xs] = (255, 255, 255)
                except Exception:
                    pass
            mask_drawn = True
        except Exception:
            pass
    if not mask_drawn:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    return annotated


# ────────────────────────────────────────────────────────────────
#  Pipeline comportement : instances singletons partagées par la boucle.
#  Les state dicts sont référencés depuis STATE — mêmes objets, pas de copie.
# ────────────────────────────────────────────────────────────────
_journal = EventJournal(events_ref=STATE["events"])
_analyzer = BehaviorAnalyzer(
    track_history=STATE["track_history"],
    track_names=STATE.setdefault("_track_names", {}),
    head_down_flags=STATE.setdefault("_head_down", {}),
    lying_flags=STATE.setdefault("_lying", {}),
)
_isolation = IsolationMonitor(journal=_journal)
_transitions = TransitionMonitor(journal=_journal)
# Posture : detection tete-au-sol depuis le masque + persistance temporelle
_head_analyzer = MaskHeadAnalyzer()
_head_motion = HeadMotionTracker()
# Classifieur de posture appris (posture_clf.pkl). Absent au premier lancement :
# _posture_model.ok == False et tout le pipeline reste sur les regles.
_posture_model = PostureModel()

# Comportement vidéo (R(2+1)D-18 entraîné sur CVB). Optionnel : si
# behavior_video.pt est absent, _video_behavior.available == False et rien
# ne change. En CPU par sécurité (la conv 3D plante sur MPS) + inférence
# espacée (every=16) pour ne pas bloquer la boucle vidéo.
_video_behavior = VideoBehavior(device="cpu", every=16)


def analyze_behavior(boxes, track_ids, t_now, frame_shape=None):
    """Wrapper stable pour callsites existants (bench_perf, boucle principale)."""
    # Re-sync : STATE dicts peuvent être remplacés (reset_for_new_source)
    _analyzer.track_history = STATE["track_history"]
    _analyzer.track_names = STATE.get("_track_names", {})
    _analyzer.head_down_flags = STATE.get("_head_down", {})
    _analyzer.lying_flags = STATE.get("_lying", {})
    return _analyzer.analyze(boxes, track_ids, t_now, frame_shape)


def push_user_event(kind: str, text: str, name: str | None = None) -> None:
    _journal.events_ref = STATE["events"]
    _journal.push(kind, text, name)


def emit_isolation_alerts(boxes, track_ids, frame_shape) -> None:
    _isolation.journal.events_ref = STATE["events"]
    _isolation.update(
        boxes, track_ids, frame_shape,
        track_names=STATE.get("_track_names", {}),
    )


def emit_behavior_transitions(behaviors: list[dict]) -> None:
    _transitions.journal.events_ref = STATE["events"]
    _transitions.update(behaviors)


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


def db_path_for_source(source) -> str:
    """Génère un chemin de DB (.pkl) stable pour une source vidéo donnée.

    Chaque vidéo a sa propre DB, donc revenir à une vidéo déjà vue restaure
    exactement les mêmes bovins et les mêmes noms.

    Exemples :
      '107414-678258609_medium.mp4' -> 'cattle_db_107414-678258609_medium.pkl'
      0 (webcam)                   -> 'cattle_db_webcam0.pkl'
    """
    import hashlib
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return f"cattle_db_webcam{int(source)}.pkl"
    # Fichier : utilise le nom de fichier (sans extension) comme clé
    name = str(source).rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    # Enleve les timestamps numeriques en prefixe
    if "_" in name:
        parts = name.split("_", 1)
        if parts[0].isdigit() and len(parts[0]) >= 10:
            name = parts[1]
    # Enleve l'extension
    name = name.rsplit(".", 1)[0]
    # Limite la longueur pour eviter les chemins trop longs
    if len(name) > 50:
        # Fallback : hash court si le nom est trop long
        h = hashlib.md5(str(source).encode()).hexdigest()[:10]
        name = f"video_{h}"
    return f"cattle_db_{name}.pkl"


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
    # Classes COCO acceptees comme bovin (cf. detector.CATTLE_CLASS_IDS).
    extra_classes = not getattr(args, "no_extra_classes", False)
    if extra_classes:
        info("[Init] Classes acceptees: cow + horse + sheep "
             "(COCO confond ces classes sur les vues de troupeau). "
             "--no-extra-classes pour n'accepter que 'cow'.")

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
        detector = CattleDetectorMLX(
            model_name=mlx_model, device="mlx", extra_classes=extra_classes,
        )
    else:
        detector = CattleDetector(
            model_name=args.yolo_model, device=device,
            extra_classes=extra_classes,
        )
    reid = CattleReID(model_name=args.reid_model, device=reid_device)
    # DB PAR VIDÉO : utilise le chemin de DB correspondant à la source
    # initiale, pas un fichier générique. Chaque vidéo a sa propre DB.
    initial_db_path = db_path_for_source(source)
    db = EmbeddingDatabase(path=initial_db_path, reid_engine=reid)
    # Reference globale pour permettre le switch de DB au changement de video.
    STATE["db"] = db
    STATE["_reid_engine"] = reid  # necessaire pour recharger avec match vectorisé
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
            kind = getattr(engine, "_kind", "vlm").upper()
            ok(f"[Breed] {kind} zero-shot pret ({len(engine.race_names)} races)")
        else:
            warn("[Breed] modele vision-langage indisponible, fallback HSV active")
    except Exception as e:
        warn(f"[Breed] pre-charge modele echoue ({e}), fallback HSV")

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
    # Nombre de frames ou chaque piste a ete vue (garde d'age anti-fantome).
    track_age: dict[int, int] = {}
    # Dernier embedding connu par piste : sert au classifieur de posture appris,
    # qui doit pouvoir statuer meme sur les frames ou aucun embedding n'a ete
    # recalcule (le re-embedding n'a lieu que toutes les `embed_every` frames).
    track_last_emb: dict[int, np.ndarray] = {}
    # Similarite a laquelle chaque piste a revendique son nom. Sert d'arbitre
    # quand deux pistes revendiquent le meme bovin dans une frame.
    track_claim_sim: dict[int, float] = {}
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
                    new_det = CattleDetectorMLX(
                        model_name=d, device="mlx", extra_classes=extra_classes,
                    )
                else:
                    # Convertit le device MLX en device PyTorch compatible (mps/cpu)
                    pt_device = resolve_device("auto", for_pytorch=True)
                    new_det = CattleDetector(
                        model_name=d, device=pt_device, half=False,
                        extra_classes=extra_classes,
                    )
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
                        track_age.clear()
                        track_last_emb.clear()
                        STATE["_track_names"] = track_id_to_name
                        reset_for_new_source()
                        # DB PAR VIDÉO : au lieu de vider la DB, on recharge
                        # celle qui correspond à la nouvelle source. Chaque
                        # vidéo a son propre fichier .pkl, donc revenir à une
                        # vidéo déjà vue restaure exactement les mêmes bovins
                        # et les mêmes noms. Le compteur global garantit que
                        # les NOUVELLES vidéos créent des noms inédits.
                        db = STATE.get("db")
                        if db is not None:
                            new_db_path = db_path_for_source(new_src)
                            # Sauvegarde la DB actuelle avant de switcher
                            try:
                                db.save()
                            except Exception:
                                pass
                            # Recharge la DB de la nouvelle source
                            db.path = new_db_path
                            db.animals = {}
                            db._dirty = True
                            db.load()
                            db.reid_engine = STATE.get("_reid_engine")
                            name_gen.__init__(db.animals)
                            STATE["name_gen"] = name_gen
                            ok(f"[Switch] DB: {new_db_path} "
                               f"({len(db.animals)} bovins connus)")
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
                    track_age.clear()
                    track_last_emb.clear()
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
                        track_age.clear()
                        track_last_emb.clear()
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
                    current_tids = {int(t) for t in track_ids}
                    frame_names = {
                        track_id_to_name[t]
                        for t in current_tids
                        if t in track_id_to_name and track_id_to_name[t] != "?"
                    }
                    # Fraction de chaque bovin masquée par un congénère. Une
                    # imagette trop recouverte contient l'autre animal : elle
                    # ferait dériver l'empreinte de référence si on l'utilisait.
                    occl_frac = overlap_fractions(boxes)

                    # Phase 1: collecte des crops éligibles à l'embedding
                    # (1 seul forward DINOv2 batché au lieu de N forwards)
                    reid_every = max(1, int(STATE.get("embed_every_current", 30)))
                    new_track_indices: list[int] = []     # nouveaux tracks
                    reembed_track_indices: list[int] = []  # tracks existants (EMA)
                    # Clé STABLE = track_id (int). det_idx change à chaque frame
                    # et ne peut pas servir de clé pour le worker asynchrone.
                    crops_to_submit: dict[int, np.ndarray] = {}  # {track_id: crop}
                    det_idx_to_tid: dict[int, int] = {}
                    # Comportement vidéo courant par piste (rempli ci-dessous)
                    video_beh_by_tid: dict[int, str] = {}

                    for det_idx, (box, tid, conf) in enumerate(zip(boxes, track_ids, confs)):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if (x2 - x1) < MIN_CROP_SIDE_PX or (y2 - y1) < MIN_CROP_SIDE_PX:
                            continue
                        det_idx_to_tid[det_idx] = int(tid)
                        track_age[int(tid)] = track_age.get(int(tid), 0) + 1
                        crop = frame[y1:y2, x1:x2]
                        # Nourrit le buffer 16 frames du modèle vidéo et récupère
                        # le comportement (grazing, walking...) si disponible.
                        _vb = _video_behavior.update(int(tid), crop)
                        if _vb:
                            video_beh_by_tid[int(tid)] = _vb
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
                        # Soumet les crops au worker, clé = track_id (non-bloquant).
                        # Les pistes encore anonymes passent devant : leur
                        # embedding conditionne l'affichage d'un nom, alors
                        # qu'un re-embedding EMA peut attendre une frame.
                        reid_worker.submit_batch(
                            crops_to_submit,
                            priority_ids={
                                det_idx_to_tid[i] for i in new_track_indices
                                if i in det_idx_to_tid
                            },
                        )
                        # Récupère ce qui est prêt (non-bloquant, peut être vide)
                        emb_by_tid = reid_worker.collect_ready()

                    # Cache du dernier embedding par piste : consommé plus bas
                    # par le classifieur de posture appris, qui doit statuer meme
                    # sur les frames sans re-embedding.
                    for _tid_e, _emb_e in emb_by_tid.items():
                        track_last_emb[int(_tid_e)] = _emb_e

                    # Phase 3: matching / EMA / annotation
                    for det_idx, (box, tid, conf) in enumerate(zip(boxes, track_ids, confs)):
                        x1, y1, x2, y2 = map(int, box)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        if (x2 - x1) < MIN_CROP_SIDE_PX or (y2 - y1) < MIN_CROP_SIDE_PX:
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
                                # ── Arbitrage du nom ────────────────────────
                                # On cherche d'abord le MEILLEUR candidat, sans
                                # exclusion. Si ce nom est deja porte dans la
                                # frame, on compare les deux revendications au
                                # lieu de laisser le premier arrive le garder.
                                #
                                # Cas reel corrige : un bovin masque perd sa
                                # piste ; l'occulteur herite de son identifiant
                                # (tracker IoU glouton) donc de son nom. Quand
                                # le vrai bovin reapparait, son propre nom est
                                # "deja pris" -> il etait exclu et recevait un
                                # nouveau nom, definitivement ecrit en base.
                                name, sim = db.match(emb, threshold=eff_threshold)
                                if name is not None and name in frame_names:
                                    holder = next(
                                        (t for t in current_tids
                                         if track_id_to_name.get(t) == name
                                         and t != tid_int),
                                        None,
                                    )
                                    holder_sim = track_claim_sim.get(holder, 0.0)
                                    if sim > holder_sim + REASSIGN_MARGIN:
                                        # Le nouveau venu revendique nettement
                                        # mieux : on lui transfere le nom et on
                                        # force l'ancien porteur a se re-identifier.
                                        if holder is not None:
                                            track_id_to_name[holder] = "?"
                                            track_claim_sim.pop(holder, None)
                                            track_emb_accum.pop(holder, None)
                                        frame_names.discard(name)
                                        dbg(f"[Re-ID] {name} transfere piste "
                                            f"{holder}->{tid_int} "
                                            f"(sim {holder_sim:.2f}->{sim:.2f})")
                                    else:
                                        # Le porteur actuel reste legitime :
                                        # on cherche un autre candidat.
                                        name, sim = db.match(
                                            emb,
                                            threshold=eff_threshold,
                                            exclude=frame_names,
                                        )
                                if name is None:
                                    # Seconde chance au seuil abaisse, limitee
                                    # aux bovins hors champ : un animal deja
                                    # connu revu sous un autre angle matche
                                    # souvent juste sous le seuil ; le
                                    # rattacher evite de creer un doublon.
                                    relax = max(0.30, eff_threshold - SECOND_CHANCE_DELTA)
                                    name, sim = db.match(
                                        emb, threshold=relax, exclude=frame_names,
                                    )
                                event = None
                                if name is None and (
                                        track_age.get(tid_int, 0)
                                        < MIN_NEW_ID_AGE_FRAMES):
                                    # Piste trop jeune pour creer une identite :
                                    # les pistes ephemeres (artefact, doublon)
                                    # creaient chacune un bovin fantome en base
                                    # -> sur-comptage. Elle reste "?" et sera
                                    # re-tentee aux frames suivantes.
                                    track_id_to_name[tid_int] = "?"
                                    STATE["_track_names"] = track_id_to_name
                                elif name is None:
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
                                    race_txt = (
                                        f" ({coat_type})"
                                        if coat_type and coat_type != "Indeterminee"
                                        else ""
                                    )
                                    event = {
                                        "kind": "arrival",
                                        "name": proper_name,
                                        "text": f"{proper_name}{race_txt} rejoint le troupeau",
                                        "ts": time.time(),
                                    }
                                else:
                                    proper_name = name_gen.get(name)
                                    event = {
                                        "kind": "return",
                                        "name": proper_name,
                                        "text": f"{proper_name} de retour dans le champ",
                                        "ts": time.time(),
                                    }
                                if name is not None:
                                    track_id_to_name[int(tid)] = name
                                    track_emb_accum[int(tid)] = [emb]
                                    track_claim_sim[tid_int] = float(sim)
                                    STATE["_track_names"] = track_id_to_name
                                    frame_names.add(name)
                                if event:
                                    push_user_event(
                                        event["kind"], event["text"],
                                        name=event.get("name"),
                                    )
                            else:
                                track_id_to_name[int(tid)] = "?"
                                STATE["_track_names"] = track_id_to_name
                        elif det_idx in reembed_track_indices:
                            emb = emb_by_tid.get(tid_int)
                            # Imagette polluee par un congenere -> on ne met pas
                            # a jour l'empreinte de reference. Sans ce garde-fou,
                            # l'empreinte d'un bovin partiellement masque derive
                            # vers celle de l'occulteur et il finit par ne plus
                            # se reconnaitre lui-meme.
                            if (emb is not None and det_idx < len(occl_frac)
                                    and occl_frac[det_idx] > OCCLUSION_SKIP_FRAC):
                                emb = None
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

                        # Lookup du comportement courant pour ce tid
                        behavior_label = ""
                        for _b in STATE.get("behavior", []):
                            if _b.get("track_id") == int(tid):
                                behavior_label = _b.get("action", "")
                                break

                        # Comportement vidéo (R(2+1)D) pour ce tid, si dispo
                        video_beh = video_beh_by_tid.get(int(tid), "")

                        # Label affiche au-dessus de la tete : nom + comportement
                        label = f"{display_name}  {behavior_label}  {video_beh}".strip()
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        ly1 = max(0, y1 - th - 10)
                        ly2 = ly1 + th + 10
                        cv2.rectangle(annotated, (x1, ly1), (x1 + tw + 8, ly2), color, -1)
                        cv2.putText(annotated, label, (x1 + 4, ly1 + th + 2),
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
                                    source=STATE.get("source_label", "inconnu"),
                                ))
                            except Exception:
                                pass

                    # Comportement (sur la dernière frame traitée)
                    # Head-down : voir posture.MaskHeadAnalyzer. Le signal est la
                    # largeur de la plus longue plage de masque AU RAS DU SOL
                    # (sabot etroit vs mufle large), invariant a l'orientation —
                    # contrairement a l'aspect ratio qui echoue en vue laterale.
                    # Le verdict est ensuite lisse temporellement par tid.
                    head_down_by_tid = {}
                    lying_by_tid = {}
                    _active_tids = set()
                    if masks_data is not None:
                        _H, _W = frame.shape[:2]
                        for _i, _tid in enumerate(track_ids):
                            if _i >= len(masks_data):
                                continue
                            _tid_i = int(_tid)
                            _active_tids.add(_tid_i)
                            try:
                                _mr = cv2.resize(masks_data[_i], (_W, _H),
                                                 interpolation=cv2.INTER_NEAREST)
                                _x1, _y1, _x2, _y2 = boxes[_i].astype(int)
                                _roi = (_mr > 0.5)[max(0, _y1):min(_H, _y2),
                                                   max(0, _x1):min(_W, _x2)]
                                # Bas de silhouette non observable : boite
                                # coupee par le bord de l'image, ou animal
                                # largement masque par un congenere. Dans les
                                # deux cas la bande basse est pleine sans que
                                # l'animal soit couche -> on l'indique a
                                # l'analyseur pour qu'il s'abstienne.
                                _truncated = (
                                    _y2 >= _H - BOTTOM_EDGE_MARGIN_PX
                                    or (_i < len(occl_frac)
                                        and occl_frac[_i] > OCCLUSION_SKIP_FRAC)
                                )
                                _state = _head_analyzer.analyze(
                                    _roi, truncated_bottom=bool(_truncated),
                                )
                                head_down_by_tid[_tid_i] = _head_motion.update(
                                    _tid_i, _state,
                                )
                                lying_by_tid[_tid_i] = _head_motion.is_lying(_tid_i)

                                # Classifieur appris : prime sur les regles UNIQUEMENT
                                # s'il est charge, active, et suffisamment sur (sinon
                                # predict renvoie None et on garde le verdict ci-dessus).
                                # Corrige les cas ou la geometrie du masque echoue
                                # (bovin noir sur sol sombre, vue de face...).
                                if (_posture_model.ok
                                        and STATE.get("use_posture_model", True)):
                                    _pred = _posture_model.predict(
                                        track_last_emb.get(_tid_i)
                                    )
                                    if _pred is not None:
                                        lying_by_tid[_tid_i], head_down_by_tid[_tid_i] = _pred
                            except Exception:
                                pass
                        _head_motion.prune(_active_tids)
                    STATE["_head_down"] = head_down_by_tid
                    STATE["_lying"] = lying_by_tid
                    STATE["behavior"] = analyze_behavior(boxes, track_ids, time.time(), frame.shape)
                    emit_behavior_transitions(STATE["behavior"])
                    emit_isolation_alerts(boxes, track_ids, frame.shape)

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