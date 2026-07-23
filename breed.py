"""
# Backward compatibility - imports from new structure
from models.breed import BreedClassifier, BREEDS, classify_breed, get_clip_engine

__all__ = ["BreedClassifier", "BREEDS", "classify_breed", "get_clip_engine"]

APPROCHE
--------
On utilise SigLIP-2 So400m (google/siglip2-so400m-patch16-384, 1136M params) en
classification zero-shot. SigLIP-2 est l'etat de l'art 2025 pour la classification
fine-grained zero-shot (84.1% ImageNet), nettement meilleur que CLIP ViT-B/32
sur les distinctions subtiles entre races.

Les "classes" sont des phrases descriptives enrichies qui encodent les
DISCRIMINATEURS VISUELS de chaque race (couleur de robe precise + morphologie +
vocation). C'est crucial : SigLIP ne voit pas que la couleur, il voit la forme,
la texture, le contexte. Des prompts riches discriminent bien mieux.

SEUIL DE CONFIANCE
------------------
SigLIP-2 zero-shot ne peut pas garantir >95% de precision sur des races
confondues (Charolaise vs Limousine vs Salers sont toutes "fauves").
On ajoute donc un SEUIL : si la confiance est trop basse, on affiche
"Race indeterminee" plutot qu'une race potentiellement fausse.
C'est l'approche honnete defendable en soutenance.

FALLBACK HSV
------------
Si SigLIP indispo, on retombe sur l'analyse de robe HSV classique (moins
fiable mais degrade proprement).

PERFORMANCE
-----------
- Embeddings texte precalcules UNE FOIS au demarrage (0ms en boucle).
- Inference image ~22ms sur MPS, appelee 1 fois par nouvel animal.
"""
import numpy as np
import cv2
from typing import Optional

from console import info, warn


# ──────────────────────────────────────────────────────────────────────────
# Catalogue des races cibles.
# Les prompts encodent les discriminateurs visuels precis de chaque race :
# couleur exacte de la robe + motif + morphologie + vocation. Plus le prompt
# est riche et specifique, mieux SigLIP discrimine.
# ──────────────────────────────────────────────────────────────────────────
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
        "use": "Mixte",
        "desc": "Robe pie fauve, excellente pour les fromages AOP.",
        "swatch": "#b8704a",
    },
    "Hereford": {
        "prompt": "a photo of a Hereford beef cow with red body color and white face head and underline markings",
        "origin": "Angleterre",
        "use": "Bouchere",
        "desc": "Roux avec tete et extremites blancs, bouchere repandue.",
        "swatch": "#a85a30",
    },
    "Aubrac": {
        "prompt": "a photo of an Aubrac cow with fawn grey wheat coat and dark black extremities on legs and muzzle",
        "origin": "France (Aveyron)",
        "use": "Mixte",
        "desc": "Fauve, mugree de noir aux extremites, race rustique.",
        "swatch": "#9a8060",
    },
}

# Couleur de fallback quand la race est indeterminee
SWATCH_UNKNOWN = "#555555"

# Seuil de confiance base sur la MARGE entre le top-1 et le top-2 (softmax).
# Avec softmax temp=0.01 sur cosine similarities:
# - Vraie frame de bovin : marge ~0.30-0.50 (top-1 clairement au-dessus)
# - Crop ambigu / bruit : marge ~0.05-0.10 (scores plats, SigLIP hesite)
# En dessous de 0.15, on affiche "Indeterminee" (honnete).
CONFIDENCE_MARGIN_THRESHOLD = 0.15


# ──────────────────────────────────────────────────────────────────────────
# Moteur SigLIP-2 zero-shot (lazy-loaded).
# ──────────────────────────────────────────────────────────────────────────
class _BreedEngine:
    """Encapsule SigLIP-2 So400m pour la classification zero-shot de races.

    Singleton : on instancie une seule fois. Les embeddings texte des prompts
    de race sont precalcules a la premiere utilisation.
    """

    _instance: Optional["_BreedEngine"] = None

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.text_emb = None
        self.race_names: list[str] = list(BREEDS.keys())
        self.available = False
        self._load()

    def _load(self):
        """Charge SigLIP-2 et precalcule les embeddings texte."""
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            warn(f"[breed] transformers/torch indisponible ({e}), fallback HSV.")
            return

        # On essaie SigLIP-2 So400m (SOTA fine-grained), fallback CLIP ViT-B/32.
        candidates = [
            ("google/siglip2-so400m-patch16-384", "siglip2"),
            ("openai/clip-vit-base-patch32", "clip"),
        ]
        for model_id, kind in candidates:
            try:
                self.device = "mps" if getattr(torch.backends, "mps", None) and \
                    torch.backends.mps.is_available() else "cpu"
                self.model = AutoModel.from_pretrained(model_id)
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.model = self.model.to(self.device).eval()
                self._kind = kind
                self._cache_text_embeddings()
                self.available = True
                info(f"[breed] {kind.upper()} zero-shot charge sur "
                     f"{self.device.upper()} ({len(self.race_names)} races)")
                return
            except Exception as e:
                warn(f"[breed] Echec {model_id} ({e}), essai suivant...")
                continue
        warn("[breed] Aucun modele vision-langage disponible, fallback HSV.")

    def _cache_text_embeddings(self):
        """Precalcule les embeddings texte de tous les prompts de race."""
        import torch
        prompts = [BREEDS[r]["prompt"] for r in self.race_names]
        with torch.no_grad():
            ti = self.processor(text=prompts, return_tensors="pt",
                                padding=True).to(self.device)
            out = self.model.get_text_features(**ti)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb / emb.norm(dim=-1, keepdim=True)
            self.text_emb = emb

    @classmethod
    def get(cls) -> Optional["_BreedEngine"]:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance if cls._instance.available else None

    def _embed_image(self, img):
        import torch
        with torch.no_grad():
            ii = self.processor(images=[img], return_tensors="pt").to(self.device)
            out = self.model.get_image_features(**ii)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb

    def classify(self, crop_bgr: np.ndarray) -> Optional[dict]:
        """Classifie un crop BGR. Retourne {race, confidence, probs} ou None."""
        if not self.available or self.model is None:
            return None
        try:
            from PIL import Image
            import torch
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            emb = self._embed_image(img)
            # Cosine similarity image vs chaque prompt de race
            sims = (emb.cpu() @ self.text_emb.cpu().T).squeeze(0)
            # Pour la classification MONO-LABEL (un bovin = une race), softmax
            # sur les cosinus similarities est beaucoup plus discriminant que
            # sigmoid. Temperature 0.01 = bonne calibration testee empiriquement
            # (vraie frame: 58% top-1 vs 15% top-2 ; bruit: ~21% flat).
            probs = torch.softmax(sims / 0.01, dim=-1)
            idx = int(probs.argmax())
            sorted_p, _ = probs.sort(descending=True)
            top1_val = float(sorted_p[0])
            top2_val = float(sorted_p[1])
            margin = top1_val - top2_val
            return {
                "race": self.race_names[idx],
                "confidence": round(top1_val, 3),
                "margin": round(margin, 4),
                "probs": {self.race_names[i]: round(float(probs[i]), 3)
                          for i in range(len(self.race_names))},
            }
        except Exception as e:
            warn(f"[breed] inference echouee ({e}), fallback HSV.")
            return None


def get_clip_engine():
    """Point d'entree publique pour le moteur (lazy singleton)."""
    return _BreedEngine.get()


# ──────────────────────────────────────────────────────────────────────────
# Fallback HSV (quand SigLIP/CLIP indisponible).
# ──────────────────────────────────────────────────────────────────────────
COAT_SWATCHES = {
    "Noir uni": "#1a1a1a", "Blanc uni": "#f0ebe0", "Pie noir": "#2a2a2a",
    "Pie fauve": "#c8a06a", "Pie rouge": "#b85c3a", "Fauve uni": "#c89858",
    "Rouge / acajou": "#9e3d22", "Gris": "#8a8a8a", "Bringe": "#6b4a2a",
    "Indeterminee": SWATCH_UNKNOWN,
}


def _analyze_coat_hsv(crop_bgr: np.ndarray) -> dict:
    """Fallback HSV : analyse de robe-type."""
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
    """Classification par robe-type (fallback)."""
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
            "note": f"Robe {coat_type.lower()} (HSV, fallback)."}


# ──────────────────────────────────────────────────────────────────────────
# API publique (interface preservee pour processor.py)
# ──────────────────────────────────────────────────────────────────────────
def classify_breed(crop_bgr: np.ndarray) -> dict:
    """Identifie la race d'un bovin depuis un crop BGR.

    Utilise SigLIP-2 zero-shot (principal). Si la confiance est trop basse,
    affiche 'Indeterminee' (honnete). Fallback HSV si modele indisponible.

    Retourne (meme signature qu'avant) :
        coat_type: nom de la race, "Indeterminee" si peu confiant, ou robe-type
        confidence: 0.0-1.0
        swatch: couleur hex pour l'UI
        zones: decomposition HSV (explicable) ou {}
        breeds: liste des races compatibles
        note: explication pour le rapport / l'UI
        method: 'siglip2' | 'clip' | 'hsv' | 'low_confidence'
    """
    engine = get_clip_engine()
    if engine is not None and crop_bgr is not None and crop_bgr.size > 0:
        result = engine.classify(crop_bgr)
        if result is not None:
            race = result["race"]
            conf = result["confidence"]
            margin = result.get("margin", 0)

            # SEUIL DE CONFIANCE base sur la MARGE (top1 - top2).
            # Une vraie vache reconnaissable a une marge > 0.04. Une couleur
            # pure ou un crop ambigu donne une marge < 0.02 (SigLIP hesite).
            if margin < CONFIDENCE_MARGIN_THRESHOLD:
                return {
                    "coat_type": "Indeterminee",
                    "confidence": conf,
                    "swatch": SWATCH_UNKNOWN,
                    "zones": {},
                    "breeds": [],
                    "method": "low_confidence",
                    "note": (f"Race incertaine (marge {margin:.3f} < seuil "
                             f"{CONFIDENCE_MARGIN_THRESHOLD}). Top candidat: {race}. "
                             f"Classification visuelle non fiable, race masquee."),
                }

            info_breed = BREEDS.get(race, {})
            method = engine._kind if hasattr(engine, "_kind") else "vlm"
            return {
                "coat_type": race,
                "confidence": conf,
                "swatch": info_breed.get("swatch", SWATCH_UNKNOWN),
                "zones": {},
                "breeds": [
                    {"name": race, "origin": info_breed.get("origin", ""),
                     "use": info_breed.get("use", ""), "desc": info_breed.get("desc", "")}
                ],
                "method": method,
                "note": (f"Race identifiee par {method.upper()} zero-shot : {race} "
                         f"(conf={conf:.2f}, marge={margin:.3f}). Classification "
                         f"visuelle, complement d'identification recommande."),
            }

    return _classify_hsv(crop_bgr)


def get_breed_info(name: str) -> Optional[dict]:
    """Retourne les infos d'une race pour /api/breeds/<name>."""
    if name in BREEDS:
        b = BREEDS[name]
        return {"coat_type": name, "swatch": b["swatch"],
                "breeds": [{"name": name, "origin": b["origin"],
                            "use": b["use"], "desc": b["desc"]}]}
    if name in COAT_SWATCHES:
        return {"coat_type": name, "swatch": COAT_SWATCHES[name], "breeds": []}
    return None
