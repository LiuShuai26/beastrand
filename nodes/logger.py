# logger.py
from __future__ import annotations
import atexit, signal, time, sys
from multiprocessing import get_context
from dataclasses import dataclass
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter

# ---------- message types ----------
@dataclass
class _MsgScalar:
    run: str
    tag: str
    value: float
    step: int
    wall_time: Optional[float] = None

@dataclass
class _MsgFlush: ...
@dataclass
class _MsgStop: ...

# ---------- globals ----------
_queue = None
_proc = None
_started = False

def _safe_maxsize(requested: int) -> int:
    # macOS/POSIX caps semaphore initial value around 32767
    cap = 32767 if sys.platform == "darwin" else requested
    return min(requested, cap)

def start_logger(
    logdir: str = "train_logs",
    experiment_name: str = "default",
    queue_maxsize: int = 30_000,     # <= 32767 on macOS
    flush_every: float = 2.0,
    batch_write_max: int = 10_000,
    start_method: str = "spawn",
) -> None:
    """Start one writer process and the shared queue in the PARENT process."""
    global _queue, _proc, _started
    if _started:
        return
    ctx = get_context(start_method)
    qsize = _safe_maxsize(int(queue_maxsize))
    _queue = ctx.Queue(maxsize=qsize)
    _proc = ctx.Process(
        target=_logger_process,
        args=(_queue, logdir, experiment_name, float(flush_every), int(batch_write_max)),
        daemon=True,
    )
    _proc.start()
    _started = True
    atexit.register(stop_logger)
    try:
        signal.signal(signal.SIGTERM, lambda *_: stop_logger())
        signal.signal(signal.SIGINT,  lambda *_: stop_logger())
    except Exception:
        pass

def get_logger_queue():
    """Call this in the parent and pass it to children."""
    return _queue

def child_attach_logger(shared_queue) -> None:
    """Call this at the TOP of each child process to attach to the parent’s queue."""
    global _queue, _started
    _queue = shared_queue
    _started = True   # we’re attached; don’t spawn another process

def log_scalar(run: str, tag: str, value: float, step: int,
               wall_time: Optional[float] = None,
               drop_on_full: bool = True, put_timeout: float = 0.002) -> None:
    if _queue is None:
        raise RuntimeError("Logger not attached. Call start_logger() in parent and child_attach_logger(queue) in children.")
    msg = _MsgScalar(run=run, tag=tag, value=float(value), step=int(step), wall_time=wall_time)
    try:
        _queue.put(msg, block=not drop_on_full, timeout=put_timeout)
    except Exception:
        pass  # drop on overload

def flush_logger(block: bool = True, put_timeout: float = 0.1) -> None:
    if _queue is None:
        return
    try:
        _queue.put(_MsgFlush(), block=block, timeout=put_timeout)
    except Exception:
        pass

def stop_logger(join_timeout: float = 2.5) -> None:
    global _started, _proc, _queue
    if not _started:
        return
    try:
        _queue.put_nowait(_MsgStop())
    except Exception:
        pass
    if _proc and _proc.is_alive():
        _proc.join(timeout=join_timeout)
    _started = False
    _proc = None
    _queue = None

def _logger_process(queue, logdir: str, experiment_name: str, flush_every: float, batch_write_max: int):
    import signal, time
    try: signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception: pass

    writer = SummaryWriter(log_dir=f"{logdir}/{experiment_name}", flush_secs=int(max(1, flush_every)))
    last_flush = time.time()
    stopping = False

    try:
        while True:
            batch = []
            try:
                item = queue.get(timeout=0.25)
                batch.append(item)
            except Exception:
                pass

            for _ in range(batch_write_max - 1):
                try:
                    batch.append(queue.get_nowait())
                except Exception:
                    break

            if batch:
                for m in batch:
                    if isinstance(m, _MsgStop):
                        stopping = True
                        continue
                    if isinstance(m, _MsgFlush):
                        writer.flush()
                        last_flush = time.time()
                        continue
                    if isinstance(m, _MsgScalar):
                        wt = m.wall_time if m.wall_time is not None else time.time()
                        # single writer: namespace via run/
                        namespaced_tag = f"{m.run}/{m.tag}" if m.run else m.tag
                        writer.add_scalar(namespaced_tag, float(m.value), m.step, walltime=wt)

            now = time.time()
            if now - last_flush >= flush_every:
                writer.flush()
                last_flush = now

            if stopping:
                # drain tail quickly
                while True:
                    try:
                        m = queue.get_nowait()
                    except Exception:
                        break
                    if isinstance(m, _MsgScalar):
                        writer.add_scalar(f"{m.run}/{m.tag}", float(m.value), m.step, walltime=time.time())
                break
    finally:
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass