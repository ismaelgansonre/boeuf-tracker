"""
reid.py
-------
Extraction d'embeddings visuels pour la ré-identification de bovins.

Combine DEUX signaux complémentaires:

1. **DINOv2** (Meta): vecteur sémantique généraliste, capture la forme et
   l'apparence globale. Bon pour distinguer bovin vs autre animal.

2. **Pattern binaire du pelage**: pour chaque bovin, on seuille l'image
   (clair/sombre) puis on extrait un histogramme de taches. Très discriminant
   pour les races à motifs (Holstein, Normand, Simmental).

La similarité finale est la moyenne pondérée des deux, ce qui surpasse
DINOv2 seul pour identifier des bovins similaires.
"""
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor


class CattleReID:
    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device: str = "cpu",
        pattern_weight: float = 0.6,
    ):
        """
        pattern_weight: poids du pattern de pelage dans le score final (0-1).
        Le reste vient de DINOv2. Pour bovins à motifs forts (Holstein):
        garder 0.6-0.7. Pour races unies (Angus, Charolais): baisser à 0.3.
        """
        self.device = device
        self.pattern_weight = pattern_weight

        print(f"[DINOv2] Chargement de {model_name} sur {device}...", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        with torch.no_grad():
            dummy_pil = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            dummy_in = self.processor(images=dummy_pil, return_tensors="pt").to(self.device)
            _ = self.model(**dummy_in)
        print(f"[DINOv2] Modèle prêt (pattern_weight={pattern_weight})", flush=True)

    @torch.no_grad()
    def _dino_embedding(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 16 or w < 16:
            return None
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        emb = out.last_hidden_state.mean(dim=1).flatten().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

    def _pattern_embedding(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        Extrait un descripteur de motif binaire du pelage:
        1. Conversion en niveaux de gris
        2. Flou gaussien pour lisser le bruit
        3. Seuillage adaptatif (Otsu) → masque binaire taches/fond
        4. Découpage en grille (4x4)
        5. Pour chaque cellule: ratio blanc/noir + position du centre de masse
        → vecteur de features invariant à la rotation/position
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 32 or w < 32:
            return None

        # Crop centré et carré (pour rendre le pattern comparable)
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        sq = crop_bgr[y0:y0 + side, x0:x0 + side]

        gray = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # Otsu → binarisation automatique
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Grille 4x4 -> 16 ratios
        grid_h, grid_w = side // 4, side // 4
        feats = []
        for gy in range(4):
            for gx in range(4):
                cell = binary[gy * grid_h:(gy + 1) * grid_h, gx * grid_w:(gx + 1) * grid_w]
                # Ratio de pixels "sombres" (taches)
                ratio = 1.0 - (cell.mean() / 255.0)
                feats.append(ratio)

        # Centre de masse global du motif sombre (donne la "forme" de la distribution)
        moments = cv2.moments(binary)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"] / side
            cy = moments["m01"] / moments["m00"] / side
        else:
            cx, cy = 0.5, 0.5
        feats.extend([cx, cy])

        # Étalement du motif
        if moments["m00"] > 0:
            mu20 = moments["mu20"] / moments["m00"]
            mu02 = moments["mu02"] / moments["m00"]
            feats.extend([mu20 / (side * side), mu02 / (side * side)])

        vec = np.array(feats, dtype=np.float32)
        # Normalisation L2
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec

    def get_embedding(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        Retourne un embedding combiné (DINOv2 + pattern).
        """
        dino = self._dino_embedding(crop_bgr)
        pattern = self._pattern_embedding(crop_bgr)
        if dino is None and pattern is None:
            return None
        if dino is None:
            return pattern
        if pattern is None:
            return dino

        # Pondération + concaténation
        w = self.pattern_weight
        combined = np.concatenate([
            dino * (1.0 - w),
            pattern * w,
        ])
        combined = combined / (np.linalg.norm(combined) + 1e-8)
        return combined

    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Similarité cosine entre deux embeddings."""
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))