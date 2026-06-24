"""
watcher.py
----------
Surveille les modifications des fichiers .py et relance app.py automatiquement.
Plus besoin de retourner au terminal après chaque modification.

Lancement:
    python watcher.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WATCHED_FILES = [
    "app.py",
    "processor.py",
    "reid.py",
    "detector.py",
    "database.py",
    "state.py",
    "capture.py",
    "main.py",
]
LAST_MTIME = {}


def get_mtimes():
    mtimes = {}
    for f in WATCHED_FILES:
        p = ROOT / f
        if p.exists():
            mtimes[f] = p.stat().st_mtime
    return mtimes


def main():
    global LAST_MTIME
    LAST_MTIME = get_mtimes()
    print("[Watcher] Surveillance active sur:")
    for f in WATCHED_FILES:
        print(f"  - {f}")
    print("[Watcher] Modifie un fichier .py -> relance auto")
    print("[Watcher] Ctrl+C pour quitter")

    proc = None
    try:
        while True:
            time.sleep(0.5)
            mtimes = get_mtimes()
            changed = []
            for f, mt in mtimes.items():
                if f in LAST_MTIME and mt != LAST_MTIME[f]:
                    changed.append(f)
            if changed:
                for f in changed:
                    print(f"[Watcher] {f} modifié -> relance")
                LAST_MTIME = mtimes
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                # Laisse 1s pour libérer le port
                time.sleep(1.0)
                cmd = [sys.executable, str(ROOT / "app.py")] + sys.argv[1:]
                print(f"[Watcher] Lance: {' '.join(cmd)}")
                proc = subprocess.Popen(cmd)
    except KeyboardInterrupt:
        print("\n[Watcher] Fin")
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    main()
