"""
training/download_roboflow.py
-----------------------------
Télécharge un ou plusieurs datasets Roboflow Universe au format YOLO, prêts à
être ingérés par import_external.py.

SÉCURITÉ
--------
La clé API est lue depuis la variable d'environnement ROBOFLOW_API_KEY. Elle
n'est JAMAIS écrite dans le code ni affichée. Avant d'utiliser ce script :

    export ROBOFLOW_API_KEY="ta_cle_regeneree"

(régénère la clé dans Roboflow → Settings → API Keys si elle a fuité.)

Identifier un dataset
---------------------
Depuis son URL Universe : https://universe.roboflow.com/<workspace>/<project>
Le numéro de VERSION est indiqué dans la boîte « Download Dataset » de la page
(ou dans un model_id « project/6 » → version 6).

Usage
-----
    export ROBOFLOW_API_KEY="..."
    python training/download_roboflow.py \
        --datasets arthurmessias/counting-cattle:1 \
                   cow-detection-identifier/cow-behavior-tracking:6 \
        --out external

Chaque dataset atterrit dans external/<project>/ (images/ labels/ data.yaml),
exactement la structure attendue par import_external.py.
"""
import argparse
import os
import sys


def _parse_spec(spec: str):
    """'workspace/project:version' -> (workspace, project, version:int)."""
    try:
        path, version = spec.rsplit(":", 1)
        workspace, project = path.split("/", 1)
        return workspace.strip(), project.strip(), int(version)
    except Exception:
        raise SystemExit(
            f"Spec invalide: '{spec}'. Format attendu "
            "'workspace/project:version' (ex: arthurmessias/counting-cattle:1)"
        )


def main():
    ap = argparse.ArgumentParser(description="Télécharge des datasets Roboflow.")
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="specs 'workspace/project:version'")
    ap.add_argument("--out", default="external")
    ap.add_argument("--format", default="yolov8",
                    help="format d'export (yolov8 = images/ labels/ data.yaml)")
    args = ap.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY non définie.\n"
            "  export ROBOFLOW_API_KEY=\"ta_cle\"   puis relance."
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit(
            "paquet 'roboflow' absent. Installe-le :\n"
            "  .venv/bin/python -m pip install roboflow"
        )

    os.makedirs(args.out, exist_ok=True)
    rf = Roboflow(api_key=api_key)

    for spec in args.datasets:
        workspace, project, version = _parse_spec(spec)
        dest = os.path.join(args.out, project)
        print(f"→ {workspace}/{project} v{version} → {dest}")
        try:
            proj = rf.workspace(workspace).project(project)
            proj.version(version).download(args.format, location=dest)
            print(f"  ok: {dest}")
        except Exception as e:
            print(f"  ÉCHEC ({e}). Vérifie workspace/project/version sur la "
                  f"page Universe du dataset.", file=sys.stderr)

    print("\nTéléchargement terminé. Étape suivante :")
    print(f"  .venv/bin/python training/import_external.py --root {args.out} "
          "--out training/dataset")


if __name__ == "__main__":
    main()
