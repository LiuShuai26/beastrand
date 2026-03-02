# strandbus.py
from __future__ import annotations

import os
import time
from typing import Dict, List
import zmq

class SBTimeout(Exception): ...
class SBQueueFull(Exception): ...
class SBClosed(Exception): ...

class StrandBus:
    """
    Minimal ZMQ wrapper:
      - named sockets
      - PUSH/PULL and PUB/SUB (IPC endpoints)
      - bytes + multipart
      - poller to check readability on PULL and SUB sockets
    """

    def __init__(self) -> None:
        self.ctx = zmq.Context(io_threads=1)
        self.poller = zmq.Poller()
        self.sockets: Dict[str, zmq.Socket] = {}
        self._roles: Dict[str, str] = {}   # name -> "push"|"pull"|"pub"|"sub"
        self._bound: Dict[str, bool] = {}

    # ---------- socket lifecycle ----------

    def open(self, name: str, mode: str, endpoint: str, *, bind: bool) -> None:
        """
        mode: "push"|"pull"|"pub"|"sub"
        endpoint: ipc:// path, e.g., 'ipc:///tmp/beatstrand/infer.req'
        bind: True to bind, False to connect
        """
        if name in self.sockets:
            raise ValueError(f"socket '{name}' already open")

        if not endpoint.startswith("ipc://"):
            raise ValueError("This minimal StrandBus only supports ipc:// endpoints")

        if mode not in ("push", "pull", "pub", "sub"):
            raise ValueError("mode must be 'push'|'pull'|'pub'|'sub'")

        if mode == "push":
            sock_type = zmq.PUSH
        elif mode == "pull":
            sock_type = zmq.PULL
        elif mode == "pub":
            sock_type = zmq.PUB
        else:  # sub
            sock_type = zmq.SUB

        s = self.ctx.socket(sock_type)

        # Sensible defaults
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.SNDHWM, 10_000)
        s.setsockopt(zmq.RCVHWM, 10_000)

        if mode == "push":
            s.setsockopt(zmq.IMMEDIATE, 1)  # don't queue before a connection exists

        # Ensure directory exists for ipc path if we're the binder
        if bind:
            _ensure_ipc_dir(endpoint)
            s.bind(endpoint)
        else:
            s.connect(endpoint)

        # Register readable sockets
        if mode in ("pull", "sub"):
            self.poller.register(s, zmq.POLLIN)

        # For SUB, subscribe to all messages by default
        if mode == "sub":
            s.setsockopt(zmq.SUBSCRIBE, b"")

        self.sockets[name] = s
        self._roles[name] = mode
        self._bound[name] = bind

    def close(self, name: str) -> None:
        s = self.sockets.pop(name, None)
        if s is None:
            return
        role = self._roles.pop(name, None)
        self._bound.pop(name, None)
        if role in ("pull", "sub"):
            try:
                self.poller.unregister(s)
            except Exception:
                pass
        s.close(0)

    def close_all(self) -> None:
        for name in list(self.sockets.keys()):
            self.close(name)
        self.ctx.term()

    # ---------- poll ----------

    def poll(self, timeout_ms: int = 0) -> Dict[str, int]:
        """Returns {name: zmq.POLLIN} for PULL/SUB sockets that have data."""
        if not self.sockets:
            time.sleep(timeout_ms / 1000.0)
            return {}
        events = dict(self.poller.poll(timeout=timeout_ms))
        ready: Dict[str, int] = {}
        for name, sock in self.sockets.items():
            if self._roles[name] not in ("pull", "sub"):
                continue
            if sock in events:
                ready[name] = events[sock]
        return ready

    # ---------- send / recv (bytes) ----------

    def send(self, name: str, data: bytes, *, more: bool = False) -> None:
        s = self._get(name, want=("push", "pub"))
        try:
            s.send(data, flags=zmq.SNDMORE if more else 0)
        except zmq.Again as e:
            raise SBQueueFull("send HWM reached") from e
        except zmq.ZMQError as e:
            raise SBClosed(str(e)) from e

    def recv(self, name: str, *, noblock: bool = True) -> bytes:
        s = self._get(name, want=("pull", "sub"))
        flags = zmq.NOBLOCK if noblock else 0
        try:
            return s.recv(flags=flags)
        except zmq.Again as e:
            raise e
        except zmq.ZMQError as e:
            raise SBClosed(str(e)) from e

    def recv_many(self, name: str) -> list[bytes]:
        """Block until at least one message arrives, then drain all queued ones."""
        sock = self._get(name, want=("pull", "sub"))

        first = sock.recv()   # blocking
        msgs = [first]

        while True:
            try:
                msgs.append(sock.recv(flags=zmq.NOBLOCK))
            except zmq.Again:
                break
            except zmq.ZMQError as e:
                raise SBClosed(str(e)) from e

        return msgs

    def recv_at_least(self, name: str, min_num: int) -> list[bytes]:
        """Block until at least min_num messages, then drain non-blocking."""
        sock = self._get(name, want=("pull", "sub"))

        msgs: list[bytes] = [sock.recv()]
        while len(msgs) < min_num:
            msgs.append(sock.recv())
        while True:
            try:
                msgs.append(sock.recv(flags=zmq.NOBLOCK))
            except zmq.Again:
                break
            except zmq.ZMQError as e:
                raise SBClosed(str(e)) from e
        return msgs

    def recv_at_most(self, name: str, max_num: int) -> list[bytes]:
        """Block for first message, then drain up to max_num non-blocking."""
        sock = self._get(name, want=("pull", "sub"))

        msgs: list[bytes] = [sock.recv()]

        while len(msgs) < max_num:
            try:
                msgs.append(sock.recv(flags=zmq.NOBLOCK))
            except zmq.Again:
                break
            except zmq.ZMQError as e:
                raise SBClosed(str(e)) from e
        return msgs

    # ---------- multipart ----------

    def send_multipart(self, name: str, frames: List[bytes]) -> None:
        s = self._get(name, want=("push", "pub"))
        try:
            s.send_multipart(frames)
        except zmq.Again as e:
            raise SBQueueFull("send HWM reached") from e
        except zmq.ZMQError as e:
            raise SBClosed(str(e)) from e

    def recv_multipart(self, name: str, *, noblock: bool = True) -> List[bytes]:
        s = self._get(name, want=("pull", "sub"))
        flags = zmq.NOBLOCK if noblock else 0
        try:
            return s.recv_multipart(flags=flags, copy=True)
        except zmq.Again as e:
            raise e
        except zmq.ZMQError as e:
            raise SBClosed(str(e)) from e

    # ---------- configuration ----------

    def set_hwm(self, name: str, *, snd: int | None = None, rcv: int | None = None) -> None:
        s = self._get(name, want=("push", "pull", "pub", "sub"))
        if snd is not None:
            s.setsockopt(zmq.SNDHWM, snd)
        if rcv is not None:
            s.setsockopt(zmq.RCVHWM, rcv)

    def get_socket(self, name):
        s = self.sockets.get(name)
        if s is None:
            raise SBClosed(f"socket '{name}' is not open")
        return s

    # ---------- internals ----------

    def _get(self, name: str, *, want) -> zmq.Socket:
        s = self.sockets.get(name)
        if s is None:
            raise SBClosed(f"socket '{name}' is not open")
        role = self._roles.get(name)
        if isinstance(want, tuple):
            if role not in want:
                raise ValueError(f"socket '{name}' is '{role}', expected one of {want}")
        else:
            if role != want:
                raise ValueError(f"socket '{name}' is '{role}', expected '{want}'")
        return s


def _ensure_ipc_dir(endpoint: str) -> None:
    path = endpoint[len("ipc://"):]
    if path.startswith("/"):
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, mode=0o770, exist_ok=True)
