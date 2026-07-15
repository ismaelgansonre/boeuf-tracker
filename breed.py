"""
breed.py
--------
Identification de la race bovine par CLIP zero-shot (principal) + HSV (fallback).

APPROCHE
--------
On utilise CLIP (ViT-B/32) en classification zero-shot : les "classes" sont des
phrases descriptives en langage naturel ("a photo of a charolaise beef cow,
solid white cream color"). CLIP projette image et texte dans un meme espace
vectoriel 512-dim ; la race la plus proche du crop bovin est retenue.

AVANTAGES vs HSV pur
  - Comprend forme, texture ET couleur (l'HSV ne voit que la couleur)
  - Aucune image de reference requise : tout est defini en texte
  - Ajouter/modifier une race = editer une ligne de ce fichier
  - Etat de l'art pour la classification sans entrainement

PERFORMANCE
  - Les embeddings texte des prompts sont precalcules UNE FOIS au demarrage
    (0ms en boucle).
  - L'inference image seule coute ~5ms sur MPS, et n'est appelee qu'au moment
    de la creation d'un nouvel animal (pas a chaque frame).

FALLBACK HSV
  - Si CLIP/transformers/torch indisponible, on retombe sur l'analyse de robe
    HSV classique (moins fiable mais degrade proprement).
"""
import numpy as np
import cv2
from typing import Optional

from console import info, warn


# ──────────────────────────────────────────────────────────────────────────
# Catalogue des races cibles.
# Chaque entree decrit : prompt CLIP, origine, vocation, description, et la
# couleur d'echantillon (swatch hex) pour l'UI.
# Les prompts sont riches (robe + couleur + vocation) pour maximiser la
# discrimination zero-shot de CLIP.
# ──────────────────────────────────────────────────────────────────────────
BREEDS = {
    "Holstein": {
        "prompt": "a photo of a holstein friesian dairy cow with black and white spotted coat",
        "origin": "Europe (Pays-Bas)",
        "use": "Laitiere",
        "desc": "Robe pie noir (tachetee), laitiere mondiale, grand format.",
        "swatch": "#2a2a2a",
    },
    "Charolaise": {
        "prompt": "a photo of a charolaise beef cow with solid white cream coat",
        "origin": "France (Bourgogne)",
        "use": "Bouchere",
        "desc": "Robe blanc creme uniforme, grand format, excellente bouchere.",
        "swatch": "#f0ebe0",
    },
    "Limousine": {
        "prompt": "a photo of a limousin beef cow with solid fawn golden brown coat",
        "origin": "France (Limousin)",
        "use": "Bouchere",
        "desc": "Robe froment uniforme, museau clair, bouchere rustique.",
        "swatch": "#c89858",
    },
    "Salers": {
        "prompt": "a photo of a salers cow with solid dark mahogany red coat",
        "origin": "France (Auvergne)",
        "use": "Mixte",
        "desc": "Robe acajou fonce uniforme, race rustique de montagne.",
        "swatch": "#9e3d22",
    },
    "Angus": {
        "prompt": "a photo of an angus beef cow with solid black coat and no horns",
        "origin": "Ecosse",
        "use": "Bouchere",
        "desc": "Robe noire uniforme, sans cornes, bouchere premium.",
        "swatch": "#1a1a1a",
    },
    "Normande": {
        "prompt": "a photo of a normande dairy cow white coat spotted with fawn brown patches",
        "origin": "France (Normandie)",
        "use": "Mixte",
        "desc": "Robe pie (blanc tachete de fauve), bonne laitiere fromagere.",
        "swatch": "#c8a06a",
    },
    "Blonde d'Aquitaine": {
        "prompt": "a photo of a blonde d aquitaine beef cow with light cream wheat colored coat",
        "origin": "France (Aquitaine)",
        "use": "Bouchere",
        "desc": "Robe froment clair, grand format muscle, bouchere.",
        "swatch": "#e8d8b0",
    },
    "Montbeliarde": {
        "prompt": "a photo of a montbeliarde dairy cow white coat with red brown patches",
        "origin": "France (Franche-Comte)",
        "use": "Mixte",
        "desc": "Robe pie fauve, excellente pour les fromages AOP.",
        "swatch": "#b8704a",
    },
    "Hereford": {
        "prompt": "a photo of a hereford beef cow red body with white head and underline",
        "origin": "Angleterre",
        "use": "Bouchere",
        "desc": "Roux avec tete et extremites blancs, bouchere repandue.",
        "swatch": "#a85a30",
    },
    "Aubrac": {
        "prompt": "a photo of an aubrac cow with fawn grey brown coat and dark extremities",
        "origin": "France (Aveyron)",
        "use": "Mixte",
        "desc": "Fauve, mugree de noir aux extremites, race rustique.",
        "swatch": "#9a8060",
    },
}

# Couleur de fallback quand la race est indeterminee
SWATCH_UNKNOWN = "#555555"


# ──────────────────────────────────────────────────────────────────────────
# Moteur CLIP zero-shot (lazy-loaded : seul le 1er appel paie le chargement).
# ──────────────────────────────────────────────────────────────────────────
class _CLIPBreedEngine:
    """Encapsule CLIP ViT-B/32 pour la classification zero-shot de races.

    Singleton : on instancie une seule fois. Les embeddings texte des prompts
    de race sont precalcules a la premiere utilisation.
    """

    _instance: Optional["_CLIPBreedEngine"] = None

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.text_emb = None          # (N_races, 512) precalcule
        self.race_names: list[str] = list(BREEDS.keys())
        self.available = False
        self._load()

    def _load(self):
        """Charge CLIP et precalcule les embeddings texte. Echoue proprement."""
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:
            warn(f"[breed] CLIP indisponible ({e}), fallback HSV active.")
            return

        try:
            self.device = "mps" if getattr(torch.backends, "mps", None) and \
                torch.backends.mps.is_available() else "cpu"
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32")
            self.model = self.model.to(self.device).eval()
            self._cache_text_embeddings()
            self.available = True
            info(f"[breed] CLIP zero-shot charge sur {self.device.upper()} "
                 f"({len(self.race_names)} races).")
        except Exception as e:
            warn(f"[breed] Echec chargement CLIP ({e}), fallback HSV active.")
            self.model = None

    def _cache_text_embeddings(self):
        """Precalcule les embeddings texte de tous les prompts de race."""
        import torch
        prompts = [BREEDS[r]["prompt"] for r in self.race_names]
        with torch.no_grad():
            ti = self.processor(text=prompts, return_tensors="pt",
                                padding=True).to(self.device)
            out = self.model.get_text_features(
                input_ids=ti["input_ids"],
                attention_mask=ti["attention_mask"],
            )
            # transformers 5.x : get_*_features renvoie BaseModelOutputWithPooling
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb / emb.norm(dim=-1, keepdim=True)
            self.text_emb = emb  # reste sur le device (CPU ici pour la fusion)

    @classmethod
    def get(cls) -> Optional["_CLIPBreedEngine"]:
        """Retourne l'instance unique (ou None si CLIP indisponible)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance if cls._instance.available else None

    def classify(self, crop_bgr: np.ndarray) -> Optional[dict]:
        """Classifie un crop BGR. Retourne {race, confidence, probs} ou None."""
        if not self.available or self.model is None:
            return None
        try:
            import torch
            from PIL import Image
            # BGR (OpenCV) -> RGB (PIL)
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            with torch.no_grad():
                ii = self.processor(images=[img], return_tensors="pt").to(self.device)
                out = self.model.get_image_features(pixel_values=ii["pixel_values"])
                emb = out.pooler_output if hasattr(out, "pooler_output") else out
                emb = emb / emb.norm(dim=-1, keepdim=True)
                # Cosine vs chaque prompt de race
                sims = (emb.cpu() @ self.text_emb.cpu().T).squeeze(0)
                probs = torch.softmax(sims * float(self.model.logit_scale.exp()),
                                      dim=-1)
            idx = int(probs.argmax())
            return {
                "race": self.race_names[idx],
                "confidence": round(float(probs[idx]), 3),
                "probs": {self.race_names[i]: round(float(probs[i]), 3)
                          for i in range(len(self.race_names))},
            }
        except Exception as e:
            warn(f"[breed] CLIP inference echouee ({e}), fallback HSV.")
            return None


def get_clip_engine():
    """Point d'entree publique pour le moteur CLIP (lazy singleton)."""
    return _CLIPBreedEngine.get()


# ──────────────────────────────────────────────────────────────────────────
# Fallback HSV (quand CLIP indisponible). Analyse la robe-type.
# ──────────────────────────────────────────────────────────────────────────
COAT_SWATCHES = {
    "Noir uni": "#1a1a1a", "Blanc uni": "#f0ebe0", "Pie noir": "#2a2a2a",
    "Pie fauve": "#c8a06a", "Pie rouge": "#b85c3a", "Fauve uni": "#c89858",
    "Rouge / acajou": "#9e3d22", "Gris": "#8a8a8a", "Bringe": "#6b4a2a",
    "Indeterminee": SWATCH_UNKNOWN,
}


def _analyze_coat_hsv(crop_bgr: np.ndarray) -> dict:
    """Fallback HSV : analyse de robe-type. Moins fiable que CLIP."""
    if crop_bgr is None or crop_bgr.size == 0:
        return {"dominant": "Indeterminee", "zones": {}}
    h, w = crop_bgr.shape[:2]
    if h < 32 or w < 32:
        return {"dominant": "Indeterminee", "zones": {}}

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    total = h * w
    remaining = np.ones((h, w), dtype=bool)
    zones = {}
    m = (V < 55); zones["black"] = m & remaining; remaining &= ~zones["black"]
    m = (V > 195) & (S < 45); zones["white"] = m & remaining; remaining &= ~zones["white"]
    m = (H <= 12) & (S > 85); zones["red"] = m & remaining; remaining &= ~zones["red"]
    m = (H >= 10) & (H <= 35) & (S > 30); zones["fawn"] = m & remaining; remaining &= ~zones["fawn"]
    m = (S < 35) & (V >= 55) & (V <= 195); zones["grey"] = m & remaining; remaining &= ~zones["grey"]
    zones["other"] = remaining
    zone_pct = {k: round(float(v.sum()) / total * 100, 1)
                for k, v in zones.items() if float(v.sum()) / total > 0.005}
    return {"dominant": max(zone_pct, key=zone_pct.get) if zone_pct else "Indeterminee",
            "zones": zone_pct}


def _classify_hsv(crop_bgr: np.ndarray) -> dict:
    """Classification par robe-type (fallback quand CLIP indisponible)."""
    coat = _analyze_coat_hsv(crop_bgr)
    zones = coat["zones"]
    if coat["dominant"] == "Indeterminee" or not zones:
        return {"coat_type": "Indeterminee", "confidence": 0.0, "method": "hsv",
                "swatch": SWATCH_UNKNOWN, "zones": {}, "breeds": [],
                "note": "Crop trop petit pour analyser la robe."}
    black_p = zones.get("black", 0); white_p = zones.get("white", 0)
    fawn_p = zones.get("fawn", 0); red_p = zones.get("red", 0)
    grey_p = zones.get("grey", 0)
    is_pie_bw = black_p > 15 and white_p > 15
    is_pie_fw = fawn_p > 20 and white_p > 15 and black_p < 15
    is_pie_rw = red_p > 15 and white_p > 15
    coat_type = (
        "Noir uni" if black_p > 45 else
        "Blanc uni" if white_p > 45 else
        "Pie noir" if is_pie_bw else
        "Pie fauve" if is_pie_fw else
        "Pie rouge" if is_pie_rw else
        "Rouge / acajou" if red_p > 35 else
        "Bringe" if fawn_p > 25 and black_p > 20 and white_p < 10 else
        "Gris" if grey_p > 35 else
        "Fauve uni" if fawn_p > 40 else "Indeterminee")
    top_pct = max(zones.values()) if zones else 0
    return {"coat_type": coat_type, "confidence": round(min(top_pct / 70.0, 1.0), 2),
            "method": "hsv", "swatch": COAT_SWATCHES.get(coat_type, SWATCH_UNKNOWN),
            "zones": zones, "breeds": [],
            "note": f"Robe {coat_type.lower()} (HSV, fallback sans CLIP)."}


# ──────────────────────────────────────────────────────────────────────────
# API publique (interface preservee pour processor.py)
# ──────────────────────────────────────────────────────────────────────────
def classify_breed(crop_bgr: np.ndarray) -> dict:
    """Identifie la race d'un bovin depuis un crop BGR.

    Utilise CLIP zero-shot (principal). Si CLIP indisponible, retombe sur
    l'analyse HSV de la robe.

    Retourne (meme signature qu'avant, pour ne pas casser processor.py) :
        coat_type: nom de la race (CLIP) ou robe-type (HSV fallback)
        confidence: 0.0-1.0
        swatch: couleur hex pour l'UI
        zones: decomposition HSV (explicable) ou {}
        breeds: liste des races compatibles (tjs vide en CLIP : on a la race)
        note: explication pour le rapport / l'UI
    """
    # 1. Essayer CLIP d'abord (fiable)
    engine = get_clip_engine()
    if engine is not None and crop_bgr is not None and crop_bgr.size > 0:
        result = engine.classify(crop_bgr)
        if result is not None:
            race = result["race"]
            info_breed = BREEDS.get(race, {})
            return {
                "coat_type": race,                    # cle pour le dashboard/heatmap
                "confidence": result["confidence"],
                "swatch": info_breed.get("swatch", SWATCH_UNKNOWN),
                "zones": {},                          # pas pertinent en CLIP
                "breeds": [
                    {"name": race, "origin": info_breed.get("origin", ""),
                     "use": info_breed.get("use", ""), "desc": info_breed.get("desc", "")}
                ],
                "method": "clip",
                "note": (f"Race identifiee par CLIP zero-shot : {race} "
                         f"({result['confidence']:.0%}). "
                         f"Classification visuelle, complement d'identification recommande."),
            }

    # 2. Fallback HSV
    return _classify_hsv(crop_bgr)


def get_breed_info(name: str) -> Optional[dict]:
    """Retourne les infos d'une race (ou robe-type) pour l'endpoint /api/breeds/<name>."""
    if name in BREEDS:
        b = BREEDS[name]
        return {"coat_type": name, "swatch": b["swatch"],
                "breeds": [{"name": name, "origin": b["origin"],
                            "use": b["use"], "desc": b["desc"]}]}
    if name in COAT_SWATCHES:
        return {"coat_type": name, "swatch": COAT_SWATCHES[name], "breeds": []}
    return None
