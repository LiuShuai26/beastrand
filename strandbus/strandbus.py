# strandbus.py
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Tuple
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

        if mode not in ("push", "pull", "pub", "sub", "router", "dealer"):
            raise ValueError("mode must be 'push'|'pull'|'pub'|'sub'|'router'|'dealer'")

        if mode == "push":
            sock_type = zmq.PUSH
        elif mode == "pull":
            sock_type = zmq.PULL
        elif mode == "pub":
            sock_type = zmq.PUB
        elif mode == "sub":
            sock_type = zmq.SUB
        elif mode == "router":
            sock_type = zmq.ROUTER
        else:  # dealer
            sock_type = zmq.DEALER

        s = self.ctx.socket(sock_type)

        # Sensible defaults
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.SNDHWM, 10_000)
        s.setsockopt(zmq.RCVHWM, 10_000)

        if mode in ("push", "dealer"):
            s.setsockopt(zmq.IMMEDIATE, 1)  # don't queue before a connection exists

        if mode == "router":
            # If you try to send to an unknown peer, raise immediately (helps catch bugs)
            try:
                s.setsockopt(zmq.ROUTER_MANDATORY, 1)
            except Exception as e:
                logging.error(e)

        # Ensure directory exists for ipc path if we're the binder
        if bind:
            _ensure_ipc_dir(endpoint)

        if bind:
            s.bind(endpoint)
        else:
            s.connect(endpoint)

        # Register readable sockets
        if mode in ("pull", "sub", "router", "dealer"):
            self.poller.register(s, zmq.POLLIN)

        # For SUB, default to no subscriptions; user calls set_subscribe()
        if mode == "sub":
            s.setsockopt(zmq.SUBSCRIBE, b"")

        self.sockets[name] = s
        self._roles[name] = mode
        self._bound[name] = bind

    def close(self, name: str) -> None:
        s = self.sockets.pop(name, None)
        if s is None:
            return
        if self._roles.get(name) in ("pull", "sub"):
            try:
                self.poller.unregister(s)
            except Exception:
                pass
        s.close(0)
        self._roles.pop(name, None)
        self._bound.pop(name, None)

    def close_all(self) -> None:
        for name in list(self.sockets.keys()):
            self.close(name)
        self.ctx.term()

    # ---------- SUB helpers ----------

    def set_subscribe(self, name: str, prefix: bytes = b"") -> None:
        """Set subscription prefix for a SUB socket (default empty = all)."""
        s = self._get(name, want="sub")
        # Clear all, then set
        s.setsockopt(zmq.UNSUBSCRIBE, b"")
        s.setsockopt(zmq.SUBSCRIBE, prefix)

    def set_conflate(self, name: str, enabled: bool = True) -> None:
        """Keep only the latest message in the inbound queue (useful for model broadcasts)."""
        s = self._get(name, want="sub")
        s.setsockopt(zmq.CONFLATE, 1 if enabled else 0)

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
        s = self._get(name, want=("push", "pub", "dealer"))
        try:
            s.send(data, flags=zmq.SNDMORE if more else 0)
        except zmq.Again as e:
            raise SBQueueFull("send HWM reached") from e
        except zmq.ZMQError as e:
            raise SBClosed(str(e)) from e

    def recv(self, name: str, *, noblock: bool = True) -> bytes:
        s = self._get(name, want=("pull", "sub", "dealer"))
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

        # Wait for at least one message
        first = sock.recv()   # blocking
        msgs = [first]

        # Drain the rest without blocking
        while True:
            try:
                msgs.append(sock.recv(flags=zmq.NOBLOCK))
            except zmq.Again:
                break
            except zmq.ZMQError as e:
                raise SBClosed(str(e)) from e

        return msgs

    def recv_at_least(self, name: str, min_num: int) -> list[bytes]:
        """Block until at least min_num messages have been received, then drain the rest non-blocking."""
        sock = self._get(name, want=("pull", "sub", "dealer"))

        msgs: list[bytes] = [sock.recv()]
        # Get first message (blocking)
        # Keep blocking until we reach min_num
        while len(msgs) < min_num:
            msgs.append(sock.recv())  # blocking
        # Drain any extra without blocking
        while True:
            try:
                msgs.append(sock.recv(flags=zmq.NOBLOCK))
            except zmq.Again:
                break
            except zmq.ZMQError as e:
                raise SBClosed(str(e)) from e
        return msgs

    def recv_at_most(self, name: str, max_num: int) -> list[bytes]:
        sock = self._get(name, want=("pull", "sub", "dealer"))

        # Wait for at least one message
        msgs: list[bytes] = [sock.recv()]

        # Drain the rest without blocking
        while True and len(msgs) < max_num:
            try:
                msgs.append(sock.recv(flags=zmq.NOBLOCK))
            except zmq.Again:
                break
            except zmq.ZMQError as e:
                raise SBClosed(str(e)) from e
        return msgs

    # ---------- multipart ----------

    def send_multipart(self, name: str, frames: List[bytes]) -> None:
        s = self._get(name, want=("push", "pub", "dealer"))
        try:
            s.send_multipart(frames)
        except zmq.Again as e:
            raise SBQueueFull("send HWM reached") from e
        except zmq.ZMQError as e:
            raise SBClosed(str(e)) from e

    def recv_multipart(self, name: str, *, noblock: bool = True) -> List[bytes]:
        s = self._get(name, want=("pull", "sub", "dealer", "router"))
        flags = zmq.NOBLOCK if noblock else 0
        try:
            return s.recv_multipart(flags=flags, copy=True)
        except zmq.Again as e:
            raise e
        except zmq.ZMQError as e:
            raise SBClosed(str(e)) from e

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

    def set_identity(self, name: str, ident: bytes) -> None:
        s = self._get(name, want=("dealer",))
        s.setsockopt(zmq.IDENTITY, ident)

    def set_hwm(self, name: str, *, snd: int | None = None, rcv: int | None = None) -> None:
        s = self._get(name, want=("push", "pull", "pub", "sub", "router", "dealer"))
        if snd is not None:
            s.setsockopt(zmq.SNDHWM, snd)
        if rcv is not None:
            s.setsockopt(zmq.RCVHWM, rcv)

    def set_router_mandatory(self, name: str, enabled: bool = True) -> None:
        s = self._get(name, want=("router",))
        try:
            s.setsockopt(zmq.ROUTER_MANDATORY, 1 if enabled else 0)
        except Exception:
            pass

    def router_send(self, name: str, peer_id: bytes, payload: bytes) -> None:
        """ROUTER: send payload to a specific peer."""
        s = self._get(name, want=("router",))
        try:
            s.send_multipart([peer_id, payload])
        except zmq.ZMQError as e:
            raise SBClosed(str(e)) from e

    def router_recv(self, name: str, *, noblock: bool = True) -> Tuple[bytes, bytes]:
        """ROUTER: receive one message; returns (peer_id, payload)."""
        s = self._get(name, want=("router",))
        flags = zmq.NOBLOCK if noblock else 0
        try:
            frames = s.recv_multipart(flags=flags, copy=True)
        except zmq.Again as e:
            raise e
        except zmq.ZMQError as e:
            raise SBClosed(str(e)) from e

        if len(frames) == 2:
            peer, payload = frames
        elif len(frames) == 3 and frames[1] == b"":
            peer, _, payload = frames
        else:
            raise ValueError(f"unexpected ROUTER frames: {frames!r}")
        return peer, payload

    def router_recv_many(self, name: str, *, timeout_ms: int = 1, max_items: int = 1024) -> List[Tuple[bytes, bytes]]:
        """
        Block up to timeout_ms for the first message, then drain non-blocking up to max_items.
        Returns list of (peer_id, payload).
        """
        s = self._get(name, want=("router",))
        items: List[Tuple[bytes, bytes]] = []

        # Wait for at least one
        if timeout_ms is not None and timeout_ms > 0:
            ready = self.poll(timeout_ms=timeout_ms)
            if name not in ready:
                return items  # none ready within timeout

        try:
            frames = s.recv_multipart(flags=0, copy=True)  # blocking for the first one
        except zmq.ZMQError as e:
            raise SBClosed(str(e)) from e

        # Parse first
        if len(frames) == 2:
            items.append((frames[0], frames[1]))
        elif len(frames) == 3 and frames[1] == b"":
            items.append((frames[0], frames[2]))
        else:
            raise ValueError(f"unexpected ROUTER frames: {frames!r}")

        # Drain rest
        while len(items) < max_items:
            try:
                frames = s.recv_multipart(flags=zmq.NOBLOCK, copy=True)
            except zmq.Again:
                break
            except zmq.ZMQError as e:
                raise SBClosed(str(e)) from e

            if len(frames) == 2:
                items.append((frames[0], frames[1]))
            elif len(frames) == 3 and frames[1] == b"":
                items.append((frames[0], frames[2]))
            else:
                raise ValueError(f"unexpected ROUTER frames: {frames!r}")
        return items

    def get_socket(self, name):
        s = self.sockets.get(name)
        if s is None:
            raise SBClosed(f"socket '{name}' is not open")
        return s


def _ensure_ipc_dir(endpoint: str) -> None:
    path = endpoint[len("ipc://"):]
    if path.startswith("/"):
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, mode=0o770, exist_ok=True)
