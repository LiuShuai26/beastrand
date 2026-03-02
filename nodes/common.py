import logging
import signal
import time
from typing import Dict, Optional


def child_logging_setup() -> None:
    """Configure logging in a subprocess."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(processName)s | %(message)s",
    )


def child_sig_setup() -> None:
    """Ignore SIGINT in child processes so main process controls shutdown."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass


class ProfileAccum:
    """Lightweight cumulative timer for profiling hot loops."""
    __slots__ = ("_timers", "_counts", "_last_report", "_interval")

    def __init__(self, interval: float = 5.0):
        self._timers: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}
        self._last_report = time.monotonic()
        self._interval = interval

    def add(self, name: str, dt: float) -> None:
        self._timers[name] = self._timers.get(name, 0.0) + dt
        self._counts[name] = self._counts.get(name, 0) + 1

    def maybe_report(self, tag: str) -> Optional[str]:
        now = time.monotonic()
        if now - self._last_report < self._interval:
            return None
        elapsed = now - self._last_report
        parts = []
        for k in sorted(self._timers):
            total_ms = self._timers[k] * 1000
            cnt = self._counts[k]
            avg_ms = total_ms / cnt if cnt else 0
            parts.append(f"{k}: {total_ms:.0f}ms/{cnt}calls ({avg_ms:.2f}ms avg)")
        self._timers.clear()
        self._counts.clear()
        self._last_report = now
        if parts:
            return f"[{tag}] {elapsed:.1f}s | " + ", ".join(parts)
        return None
