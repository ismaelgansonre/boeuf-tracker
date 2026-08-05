"""
training/review.py
------------------
Petit outil web LOCAL pour corriger vite les postures mal étiquetées.

Le problème : après collect_dataset.py, tes imagettes de vidéos sont rangées
selon les RÈGLES (qui se trompent, ex. vache noire couchée -> pature). Les
retrouver à la main dans un dossier de 15 000 images est impossible.

Cet outil n'affiche QUE tes crops de vidéos (noms `IMG_*`, `*_f*_t*`), en
grille, groupés par dossier courant. Chaque imagette a 3 boutons
(couché / debout / pâture) : un clic déplace le fichier dans le bon dossier,
instantanément. Tu scannes `pature/`, tu repères les vaches couchées (évident
à l'œil même en noir), tu cliques — quelques minutes au lieu d'une heure.

Les embeddings ne bougent pas : ils sont indexés par nom de fichier, et le
dossier = l'étiquette. Après correction, relance train_posture.py.

Usage
-----
    python training/review.py                 # tes crops vidéo, toutes postures
    python training/review.py --folder pature # seulement le dossier pature/
    python training/review.py --all           # inclut aussi les crops externes
    python training/review.py --port 8000
Puis ouvre http://127.0.0.1:8000 dans ton navigateur.
"""
import argparse
import html
import os
import posixpath
import shutil
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ("couché", "debout", "pature")


def is_video_crop(fname: str) -> bool:
    """Vrai pour les imagettes issues de collect_dataset (tes vidéos), pas des
    datasets externes (préfixe `ext_`)."""
    return not fname.startswith("ext_")


class Reviewer:
    def __init__(self, data_dir, only_folder, include_external):
        self.data_dir = data_dir
        self.only_folder = only_folder
        self.include_external = include_external

    def items(self):
        """Liste [(label_courant, fname)] à afficher."""
        out = []
        folders = [self.only_folder] if self.only_folder else LABELS
        for lab in folders:
            d = os.path.join(self.data_dir, lab)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                if not self.include_external and not is_video_crop(fn):
                    continue
                out.append((lab, fn))
        return out

    def move(self, fname, to_label):
        """Déplace fname vers le dossier to_label. Cherche dans tous les labels."""
        if to_label not in LABELS:
            return False
        for lab in LABELS:
            src = os.path.join(self.data_dir, lab, fname)
            if os.path.exists(src):
                dst = os.path.join(self.data_dir, to_label, fname)
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.move(src, dst)
                return True
        return False


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Review postures</title><style>
body{{font-family:-apple-system,sans-serif;background:#1a1a1a;color:#eee;margin:0;padding:16px}}
h1{{font-size:16px}} .info{{color:#9c9;margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px}}
.card{{background:#262626;border-radius:8px;padding:6px;border:3px solid transparent}}
.card img{{width:100%;border-radius:4px;display:block}}
.btns{{display:flex;gap:4px;margin-top:4px}}
.btns button{{flex:1;padding:5px 0;border:0;border-radius:4px;cursor:pointer;font-size:12px;background:#3a3a3a;color:#ccc}}
.cur-couché{{border-color:#5cb85c}} .cur-debout{{border-color:#428bca}} .cur-pature{{border-color:#d9a441}}
.b-active{{background:#5cb85c;color:#000;font-weight:600}}
</style></head><body>
<h1>Correction des postures — clique la bonne étiquette</h1>
<div class="info">{count} imagettes de tes vidéos. Repère les vaches COUCHÉES mal rangées et clique « couché ». Les déplacements sont sauvegardés à chaque clic.</div>
<div class="grid">{cards}</div>
<script>
function mv(el, fname, to){{
  fetch('/move?'+new URLSearchParams({{file:fname,to:to}}),{{method:'POST'}})
   .then(r=>{{ if(r.ok){{
     const card=el.closest('.card');
     card.className='card cur-'+to;
     card.querySelectorAll('button').forEach(b=>b.classList.remove('b-active'));
     el.classList.add('b-active');
   }}}});
}}
</script></body></html>"""


def render(items):
    cards = []
    for lab, fn in items:
        q = urllib.parse.quote(fn)
        btns = "".join(
            f'<button class="{"b-active" if l==lab else ""}" '
            f'onclick="mv(this,\'{html.escape(fn)}\',\'{l}\')">{l}</button>'
            for l in LABELS
        )
        cards.append(
            f'<div class="card cur-{lab}">'
            f'<img loading="lazy" src="/img?file={q}">'
            f'<div class="btns">{btns}</div></div>'
        )
    return PAGE.format(count=len(items), cards="".join(cards))


def make_handler(rev: Reviewer):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silencieux
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self._send(200, "text/html; charset=utf-8",
                           render(rev.items()).encode("utf-8"))
            elif parsed.path == "/img":
                qs = urllib.parse.parse_qs(parsed.query)
                fname = qs.get("file", [""])[0]
                # anti-traversal : uniquement un basename
                fname = posixpath.basename(fname)
                for lab in LABELS:
                    p = os.path.join(rev.data_dir, lab, fname)
                    if os.path.exists(p):
                        self._send(200, "image/jpeg", open(p, "rb").read())
                        return
                self._send(404, "text/plain", b"not found")
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/move":
                qs = urllib.parse.parse_qs(parsed.query)
                fname = posixpath.basename(qs.get("file", [""])[0])
                to = qs.get("to", [""])[0]
                ok = rev.move(fname, to)
                self._send(200 if ok else 400, "text/plain",
                           b"ok" if ok else b"fail")
            else:
                self._send(404, "text/plain", b"not found")
    return H


def main():
    ap = argparse.ArgumentParser(description="Outil de review des postures.")
    ap.add_argument("--data", default=os.path.join(_ROOT, "training", "dataset"))
    ap.add_argument("--folder", choices=LABELS, default=None,
                    help="ne montrer qu'un dossier (ex: pature)")
    ap.add_argument("--all", action="store_true",
                    help="inclure aussi les crops externes (ext_*)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    rev = Reviewer(args.data, args.folder, args.all)
    n = len(rev.items())
    if n == 0:
        print("Aucune imagette à revoir (dataset vide ? mauvais --data ?)")
        sys.exit(1)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(rev))
    print(f"{n} imagettes à revoir.", flush=True)
    print(f"Ouvre http://127.0.0.1:{args.port} dans ton navigateur.", flush=True)
    print("Ctrl+C pour arrêter (les corrections sont déjà sauvegardées).", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêté.")


if __name__ == "__main__":
    main()
