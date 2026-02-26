import logging
import signal


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
