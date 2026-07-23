"""
models/breed.py
---------------
Breed identification via SigLIP-2 zero-shot (primary) + HSV fallback.
"""
import numpy as np
import cv2
from typing import Optional

import sys
sys.path.insert(0, '..')
from utils.console import info, warn


# ─── Breed Catalog ────────────────────────────────────────────────────────────
BREEDS = {
    "Holstein": {
        "prompt": "a photo of a Holstein Friesian dairy cow with distinctive black and white spotted patches coat, large frame",
        "origin": "Europe (Pays-Bas)",
        "use": "Laitiere",
        "desc": "Robe pie noir (tachetee), laitiere mondiale, grand format.",
        "swatch": "#2a2a2a",
    },
    "Charolaise": {
        "prompt": "a photo of a Charolais beef cow with solid white cream colored coat, very large heavily muscled frame",
        "origin": "France (Bourgogne)",
        "use": "Bouchere",
        "desc": "Robe blanc creme uniforme, grand format, excellente bouchere.",
        "swatch": "#f0ebe0",
    },
    "Limousine": {
        "prompt": "a photo of a Limousin beef cow with solid golden wheat reddish fawn coat and lighter muzzle",
        "origin": "France (Limousin)",
        "use": "Bouchere",
        "desc": "Robe froment uniforme, museau clair, bouchere rustique.",
        "swatch": "#c89858",
    },
    "Salers": {
        "prompt": "a photo of a Salers cow with solid dark mahogany deep red coat, rugged mountain cattle",
        "origin": "France (Auvergne)",
        "use": "Mixte",
        "desc": "Robe acajou fonce uniforme, race rustique de montagne.",
        "swatch": "#9e3d22",
    },
    "Angus": {
        "prompt": "a photo of an Aberdeen Angus beef cow with solid black coat, polled hornless, compact muscular frame",
        "origin": "Ecosse",
        "use": "Bouchere",
        "desc": "Robe noire uniforme, sans cornes, bouchere premium.",
        "swatch": "#1a1a1a",
    },
    "Normande": {
        "prompt": "a photo of a Normande dairy cow with white coat spotted with brown fawn patches in brindled pattern",
        "origin": "France (Normandie)",
        "use": "Mixte",
        "desc": "Robe pie (blanc tachete de fauve), bonne laitiere fromagere.",
        "swatch": "#c8a06a",
    },
    "Blonde d'Aquitaine": {
        "prompt": "a photo of a Blonde d Aquitaine beef cow with light cream wheat colored coat, tall large frame",
        "origin": "France (Aquitaine)",
        "use": "Bouchere",
        "desc": "Robe froment clair, grand format muscle, bouchere.",
        "swatch": "#e8d8b0",
    },
    "Montbeliarde": {
        "prompt": "a photo of a Montbeliarde dairy cow with white coat with red brown patches piebald pattern",
        "origin": "France (Franche-Comte)",
        "use": "Laitiere",
        "desc": "Robe pie (blanc et rouge-brun), excellente laitiere.",
        "swatch": "#f5f5f5",
    },
    "Gasconne": {
        "prompt": "a photo of a Gasconne beef cow with solid dark charcoal gray coat, medium sized hardy cattle",
        "origin": "France (Pyrenees)",
        "use": "Bouchere",
        "desc": "Robe gris anthracite, race rustique du sud.",
        "swatch": "#4a4a4a",
    },
    "Aubrac": {
        "prompt": "a photo of an Aubrac beef cow with solid fawn wheat colored coat, medium frame mountain cattle",
        "origin": "France (Aubrac)",
        "use": "Bouchere",
        "desc": "Robe froment clair, race rustique du Massif Central.",
        "swatch": "#d4a86a",
    },
    "Parthenaise": {
        "prompt": "a photo of a Parthenaise beef cow with solid dark brown mouse gray coat, medium sized cattle",
        "origin": "France (Poitou)",
        "use": "Bouchere",
        "desc": "Robe gris fonce, race bouchere du Grand Ouest.",
        "swatch": "#5c4033",
    },
    "Highland": {
        "prompt": "a photo of a Highland cattle cow with long shaggy wavy coat in red brown golden or black colors, small frame",
        "origin": "Ecosse (Highlands)",
        "use": "Mixte",
        "desc": "Longue robe hirsute (diverse couleurs), petite race rustique.",
        "swatch": "#8b4513",
    },
    "Hereford": {
        "prompt": "a photo of a Hereford beef cow with dark red coat and white face white belt pattern, medium frame",
        "origin": "Angleterre",
        "use": "Bouchere",
        "desc": "Robe rouge fonce avec tete blanche et ceinture blanche.",
        "swatch": "#8b0000",
    },
    "Simmental": {
        "prompt": "a photo of a Simmental Fleckvieh dairy beef cow with white and red spotted patchy coat pattern, large frame",
        "origin": "Allemagne/Suisse",
        "use": "Mixte",
        "desc": "Robe pie rouge et blanc, grand format, mixte.",
        "swatch": "#cc2222",
    },
    "Tarine": {
        "prompt": "a photo of a Tarine cow with dark brown black coat, medium frame, from French Alps",
        "origin": "France (Alpes)",
        "use": "Laitiere",
        "desc": "Robe sombre, laitiere alpine produisant le Beaufort.",
        "swatch": "#2d1f1a",
    },
}


class BreedClassifier:
    """Breed classifier using color analysis (HSV-based fallback)."""

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Args:
            confidence_threshold: Minimum confidence to report a breed.
                                 Below this, returns "Indeterminee".
        """
        self.confidence_threshold = confidence_threshold

    def classify(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Classify breed from crop using HSV color analysis.
        Returns (breed_name, confidence).
        """
        if crop is None or crop.size == 0:
            return "Indeterminee", 0.0
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            avg_h = np.mean(hsv[:, :, 0])
            avg_s = np.mean(hsv[:, :, 1])
            avg_v = np.mean(hsv[:, :, 2])
            breed, conf = self._hsv_to_breed(avg_h, avg_s, avg_v)
            if conf < self.confidence_threshold:
                return "Indeterminee", conf
            return breed, conf
        except Exception:
            return "Indeterminee", 0.0

    def _hsv_to_breed(self, h: float, s: float, v: float) -> tuple[str, float]:
        """Map HSV averages to breed."""
        # Very rough heuristic mapping
        if v < 40:  # Very dark
            if s < 50:
                return "Highland", 0.4
            return "Angus", 0.5
        if v > 180 and s < 30:  # Very light, desaturated = white/cream
            return "Charolaise", 0.6
        if h < 15 or h > 170:  # Red tones
            if v > 140:
                return "Limousine", 0.6
            if v > 80:
                return "Salers", 0.55
            return "Aubrac", 0.4
        if 15 <= h < 40:  # Orange-yellow
            if s > 150:
                return "Blonde d'Aquitaine", 0.5
            return "Normande", 0.4
        if 40 <= h < 80:  # Yellow-green
            return "Highland", 0.35
        if 80 <= h < 130:  # Green-cyan
            if s < 60:
                return "Charolaise", 0.35
            return "Holstein", 0.4
        if 130 <= h < 170:  # Blue-purple
            return "Tarine", 0.4
        return "Indeterminee", 0.2

    def __repr__(self) -> str:
        return f"BreedClassifier(threshold={self.confidence_threshold})"


# ─── CLIP-based classifier (for when transformers is available) ───────────────
_CLIP_AVAILABLE = False
_clip_model = None
_clip_processor = None


def get_clip_engine():
    """Lazy-load CLIP model if available."""
    global _clip_model, _clip_processor, _CLIP_AVAILABLE
    if _CLIP_AVAILABLE:
        return _clip_model, _clip_processor
    try:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP_AVAILABLE = True
        info("[Breed] CLIP loaded for zero-shot classification")
        return _clip_model, _clip_processor
    except Exception:
        _CLIP_AVAILABLE = False
        return None, None


def classify_breed(crop: np.ndarray, threshold: float = 0.3) -> tuple[str, float]:
    """
    Classify breed using CLIP zero-shot if available, else HSV fallback.
    Returns (breed_name, confidence).
    """
    clip_model, clip_processor = get_clip_engine()
    if clip_model is not None:
        try:
            from PIL import Image
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            texts = [BREEDS[b]["prompt"] for b in BREEDS]
            inputs = clip_processor(text=texts, images=pil, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            best_idx = probs.argmax().item()
            best_breed = list(BREEDS.keys())[best_idx]
            best_conf = probs[best_idx].item()
            if best_conf < threshold:
                return "Indeterminee", best_conf
            return best_breed, best_conf
        except Exception as e:
            warn(f"[Breed] CLIP classification failed: {e}")
    # Fallback to HSV
    classifier = BreedClassifier(confidence_threshold=threshold)
    return classifier.classify(crop)
