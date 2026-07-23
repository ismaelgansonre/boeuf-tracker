"""
core/capture.py
---------------
Helpers for opening video sources with proper backend and hot-swapping
between webcam / file / RTSP stream.
"""
import time
import cv2

import sys
sys.path.insert(0, '..')
from utils.console import ok, warn


class CaptureManager:
    """Manages video capture with auto-recovery."""

    def __init__(self, source):
        """
        Args:
            source: int (webcam index), str (file path/URL), or CV2 VideoCapture
        """
        self.source = source
        self.cap = None
        self._open()

    def _open(self) -> bool:
        """Open the capture source."""
        if isinstance(self.source, int):
            # Try DirectShow before MSMF on Windows (more stable for webcam)
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                try:
                    cap = cv2.VideoCapture(self.source, backend)
                    if cap.isOpened():
                        ok_, _ = cap.read()
                        if ok_:
                            self.cap = cap
                            return True
                        cap.release()
                except Exception:
                    pass
            return False
        self.cap = cv2.VideoCapture(self.source)
        return self.cap.isOpened()

    def read(self) -> tuple[bool, ...]:
        """
        Read frame with auto-recovery.
        Returns (ret, frame).
        For webcam: tries to reconnect on failure.
        For file: rewinds to beginning.
        """
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        if ret:
            return ret, frame
        if isinstance(self.source, int):
            # Webcam: try to reconnect
            warn("[Webcam] Stream lost, reconnecting...")
            self.cap.release()
            time.sleep(2.0)
            if self._open():
                return False, None  # Signal that capture was replaced
            time.sleep(3.0)
            if self._open():
                return False, None
            return False, None
        # File: rewind
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return False, None

    def release(self) -> None:
        """Release the capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @property
    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    @property
    def fps(self) -> float:
        if self.cap is None:
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS) or 0.0

    @property
    def frame_count(self) -> int:
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        if self.cap is None:
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

    def __repr__(self) -> str:
        return f"CaptureManager(source={self.source}, opened={self.is_opened})"


def open_capture(src):
    """
    Opens a video source robustly.
    Returns an open VideoCapture or None.
    """
    if isinstance(src, int):
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            try:
                cap = cv2.VideoCapture(src, backend)
                if cap.isOpened():
                    ok_, _ = cap.read()
                    if ok_:
                        return cap
                    cap.release()
            except Exception:
                pass
        return None
    cap = cv2.VideoCapture(src)
    return cap if cap.isOpened() else None


def read_with_recovery(cap, src):
    """
    Read frame with auto-recovery.
    Returns (ret, frame).
    """
    ret, frame = cap.read()
    if ret:
        return ret, frame
    if isinstance(src, int):
        warn("[Webcam] Stream lost, reconnecting...")
        cap.release()
        time.sleep(2.0)
        new_cap = open_capture(src)
        if new_cap is None:
            time.sleep(3.0)
            new_cap = open_capture(src)
        if new_cap is not None:
            return False, None
        return False, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return False, None


def source_label(src) -> str:
    """Generate a human-readable label for display."""
    if isinstance(src, int):
        return f"webcam {src}"
    if isinstance(src, str):
        name = src.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        # Remove timestamp prefix "1234567890_"
        if "_" in name:
            parts = name.split("_", 1)
            if parts[0].isdigit() and len(parts[0]) >= 10:
                name = parts[1]
        return name
    return str(src)
