"""
# Backward compatibility - imports from new structure
from utils.console import Logger, log, info, ok, warn, err, dbg, evt, boot, loop, banner

__all__ = ["Logger", "log", "info", "ok", "warn", "err", "dbg", "evt", "boot", "loop", "banner"] Utilisé dans tout le projet
pour remplacer les print() bruts.

Usage:
    from console import log
    log("info", "Démarrage du tracker")
    log("warn", "FPS bas: 8.3")
    log("error", "Impossible d'ouvrir la caméra")
    log("success", "Boeuf_001 reconnu (sim=0.82)")
    log("event", "MATCH  Boeuf_003  (sim=0.71)")

Codes ANSI auto-désactivés si la sortie n'est pas un TTY
(p.ex. redirection vers fichier → pas de codes bizarres).
"""
import sys
from datetime import datetime


class _C:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"


# Détection TTY une fois pour toutes
_IS_TTY = sys.stdout.isatty()

_LEVELS = {
    "info":    ("INFO ", _C.CYAN),
    "success": (" OK  ", _C.GREEN),
    "warn":    ("WARN ", _C.YELLOW),
    "error":   ("ERR  ", _C.RED),
    "debug":   ("DBG  ", _C.GRAY),
    "event":   ("EVT  ", _C.MAGENTA),
    "boot":    ("BOOT ", _C.BOLD + _C.BLUE),
    "loop":    ("LOOP ", _C.BOLD + _C.MAGENTA),
}


def _format(level: str, msg: str) -> str:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # ms
    tag, color = _LEVELS.get(level, ("     ", _C.RESET))
    if _IS_TTY:
        return f"{_C.DIM}{ts}{_C.RESET} {color}{tag}{_C.RESET} {msg}"
    # Sans couleurs pour les logs fichiers
    return f"{ts} {tag} {msg}"


def log(level: str, msg: str) -> None:
    """Affiche un message formaté. flush=True pour les sous-processus Flask."""
    print(_format(level, msg), flush=True)


# Raccourcis pour les niveaux courants
def info(msg: str) -> None:    log("info", msg)
def ok(msg: str) -> None:      log("success", msg)
def warn(msg: str) -> None:    log("warn", msg)
def err(msg: str) -> None:     log("error", msg)
def dbg(msg: str) -> None:     log("debug", msg)
def evt(msg: str) -> None:     log("event", msg)
def boot(msg: str) -> None:    log("boot", msg)
def loop(msg: str) -> None:    log("loop", msg)


def banner(title: str, lines: list[str] | None = None) -> None:
    """Affiche un bandeau stylisé pour le démarrage."""
    bar = "═" * 64
    if _IS_TTY:
        print(f"\n{_C.BOLD}{_C.CYAN}{bar}", flush=True)
        print(f"  {title}", flush=True)
        if lines:
            for ln in lines:
                print(f"  {_C.DIM}{ln}{_C.RESET}", flush=True)
        print(f"{bar}{_C.RESET}\n", flush=True)
    else:
        print(f"\n{bar}", flush=True)
        print(f"  {title}", flush=True)
        if lines:
            for ln in lines:
                print(f"  {ln}", flush=True)
        print(f"{bar}\n", flush=True)
