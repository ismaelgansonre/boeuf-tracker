"""
training/train_posture.py
-------------------------
Étape 2 du pipeline d'apprentissage de la posture.

Lit le dataset corrigé (dossiers couché/ debout/ pature/) et entraîne un petit
classifieur sur les embeddings MegaDescriptor déjà exportés par
collect_dataset.py. L'entraînement prend quelques secondes sur M1 — ce sont des
vecteurs, pas des images.

Le modèle est sauvegardé en `posture_clf.pkl` à la racine du projet ;
posture_model.py le charge à l'exécution.

Pourquoi un MLP léger (et pas un réseau d'images)
-------------------------------------------------
L'embedding encode déjà la silhouette et l'apparence. Apprendre la posture
revient à tracer des frontières dans cet espace de 700 dimensions — un problème
que quelques centaines d'exemples et un perceptron à une couche cachée règlent,
sans GPU ni annotation massive.

Usage
-----
    python training/train_posture.py --data training/dataset
    python training/train_posture.py --data training/dataset --min-conf 0.60
"""
import argparse
import os
import pickle
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# "autre" n'est pas une posture : ces imagettes sont exclues de l'apprentissage.
TRAIN_LABELS = ("couché", "debout", "pature")
MODEL_PATH = os.path.join(_ROOT, "posture_clf.pkl")


def load_dataset(data_dir):
    """Retourne (X, y, classes). Le dossier de chaque imagette = son étiquette."""
    emb_path = os.path.join(data_dir, "embeddings.pkl")
    if not os.path.exists(emb_path):
        raise SystemExit(f"embeddings.pkl introuvable dans {data_dir} — "
                         "lance d'abord collect_dataset.py")
    with open(emb_path, "rb") as f:
        embeddings = pickle.load(f)

    X, y = [], []
    missing = 0
    for label in TRAIN_LABELS:
        d = os.path.join(data_dir, label)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if not fname.lower().endswith(".jpg"):
                continue
            emb = embeddings.get(fname)
            if emb is None:
                missing += 1
                continue
            X.append(np.asarray(emb, dtype=np.float32).ravel())
            y.append(label)
    if missing:
        print(f"[warn] {missing} imagettes sans embedding (renommées ?) ignorées")
    if not X:
        raise SystemExit("aucune imagette étiquetée exploitable")
    return np.stack(X), np.array(y)


def train(data_dir, min_conf, seed):
    X, y = load_dataset(data_dir)
    classes, counts = np.unique(y, return_counts=True)
    print("Exemples par classe :")
    for c, n in zip(classes, counts):
        print(f"  {c:8s}: {n}")
    if len(classes) < 2:
        raise SystemExit("il faut au moins deux postures différentes pour entraîner")

    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    # Split stratifié : on garde un échantillon de validation pour mesurer
    # honnêtement, sans le montrer à l'entraînement.
    strat = y if counts.min() >= 2 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=strat,
    )
    clf = MLPClassifier(
        hidden_layer_sizes=(128,), activation="relu", max_iter=800,
        early_stopping=True, random_state=seed,
    )
    clf.fit(X_tr, y_tr)

    print(f"\nPrécision validation : {clf.score(X_val, y_val):.1%}")
    print("\nRapport par classe (validation) :")
    print(classification_report(y_val, clf.predict(X_val), zero_division=0))
    print("Matrice de confusion (lignes = vérité, colonnes = prédit) :")
    print("  classes :", list(clf.classes_))
    print(confusion_matrix(y_val, clf.predict(X_val), labels=clf.classes_))

    payload = {
        "clf": clf,
        "classes": list(clf.classes_),
        "min_conf": float(min_conf),
        "dim": int(X.shape[1]),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nModèle sauvegardé : {MODEL_PATH}")
    print(f"Seuil de confiance minimal en production : {min_conf:.2f} "
          "(en dessous, l'appli retombe sur les règles géométriques)")


def main():
    ap = argparse.ArgumentParser(description="Entraîne le classifieur de posture.")
    ap.add_argument("--data", default="training/dataset")
    ap.add_argument("--min-conf", type=float, default=0.60,
                    help="confiance mini pour que le modèle prime sur les règles")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.data, args.min_conf, args.seed)


if __name__ == "__main__":
    main()
