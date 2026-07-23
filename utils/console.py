"""
utils/console.py
----------------
Colored, timestamped console logger.
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


# TTY detection once
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


class Logger:
    """Colored logger with timestamps."""

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors and _IS_TTY

    def _format(self, level: str, msg: str) -> str:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        tag, color = _LEVELS.get(level, ("     ", _C.RESET))
        if self.use_colors:
            return f"{_C.DIM}{ts}{_C.RESET} {color}{tag}{_C.RESET} {msg}"
        return f"{ts} {tag} {msg}"

    def log(self, level: str, msg: str) -> None:
        print(self._format(level, msg), flush=True)

    def info(self, msg: str) -> None:    self.log("info", msg)
    def ok(self, msg: str) -> None:      self.log("success", msg)
    def warn(self, msg: str) -> None:    self.log("warn", msg)
    def err(self, msg: str) -> None:     self.log("error", msg)
    def dbg(self, msg: str) -> None:    self.log("debug", msg)
    def evt(self, msg: str) -> None:     self.log("event", msg)
    def boot(self, msg: str) -> None:    self.log("boot", msg)
    def loop(self, msg: str) -> None:    self.log("loop", msg)

    def banner(self, title: str, lines: list | None = None) -> None:
        bar = "═" * 64
        if self.use_colors:
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


# ─── Module-level convenience functions ────────────────────────────────────────
_log = Logger()


def log(level: str, msg: str) -> None:
    _log.log(level, msg)


def info(msg: str) -> None:    _log.info(msg)
def ok(msg: str) -> None:      _log.ok(msg)
def warn(msg: str) -> None:    _log.warn(msg)
def err(msg: str) -> None:     _log.err(msg)
def dbg(msg: str) -> None:    _log.dbg(msg)
def evt(msg: str) -> None:     _log.evt(msg)
def boot(msg: str) -> None:    _log.boot(msg)
def loop(msg: str) -> None:    _log.loop(msg)


def banner(title: str, lines: list | None = None) -> None:
    _log.banner(title, lines)
