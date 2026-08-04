"""
posture_model.py
----------------
Classifieur de posture appris (couché / debout / pature), entraîné par
training/train_posture.py sur les embeddings MegaDescriptor.

C'est un AJOUT, pas un remplacement : les règles géométriques de posture.py et
behavior.py restent en place et servent de repli. Deux garde-fous :

  1. Filtrage par confiance — si le modèle n'est pas assez sûr (< min_conf),
     `predict()` renvoie None et l'appelant garde le verdict des règles.
  2. Interrupteur — STATE["use_posture_model"] (ou le flag CLI --posture-model)
     permet d'activer/désactiver à chaud, donc de comparer A/B et de revenir
     instantanément à l'ancien comportement.

Le modèle lit un vecteur DÉJÀ calculé pour la ré-identification : aucune
inférence supplémentaire à l'exécution.
"""
from __future__ import annotations

import os
import pickle

import numpy as np

from console import ok, warn

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "posture_clf.pkl")

# Traduction posture apprise → drapeaux consommés par behavior.classify().
# behavior traite `lying` en priorité, puis `head_down` ; "debout" = les deux à
# False (l'action vient alors de la vitesse : immobile / marche / court…).
_LABEL_TO_FLAGS = {
    "couché": (True, False),   # (lying, head_down)
    "pature": (False, True),
    "pâture": (False, True),   # tolère l'accent selon le nommage des dossiers
    "debout": (False, False),
}


class PostureModel:
    """Charge posture_clf.pkl si présent et expose predict(embedding)."""

    def __init__(self, path: str = _MODEL_PATH):
        self.ok = False
        self._clf = None
        self._classes: list[str] = []
        self._dim = 0
        self.min_conf = 0.60
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            self._clf = payload["clf"]
            self._classes = list(payload["classes"])
            self._dim = int(payload.get("dim", 0))
            self.min_conf = float(payload.get("min_conf", 0.60))
            self.ok = True
            ok(f"[Posture] Classifieur appris charge ({', '.join(self._classes)}, "
               f"seuil {self.min_conf:.2f})")
        except Exception as e:  # pragma: no cover - robustesse au chargement
            warn(f"[Posture] chargement du classifieur impossible ({e}), "
                 "repli sur les regles")

    def predict(self, embedding) -> tuple[bool, bool] | None:
        """Retourne (lying, head_down) si le modèle est sûr, sinon None.

        None signifie « je ne tranche pas » : l'appelant conserve le verdict des
        règles géométriques. C'est le filtrage par confiance.
        """
        if not self.ok or embedding is None:
            return None
        q = np.asarray(embedding, dtype=np.float32).ravel()
        if self._dim and q.shape[0] != self._dim:
            return None
        try:
            proba = self._clf.predict_proba(q.reshape(1, -1))[0]
        except Exception:
            return None
        idx = int(np.argmax(proba))
        if float(proba[idx]) < self.min_conf:
            return None
        label = self._classes[idx]
        return _LABEL_TO_FLAGS.get(label)
