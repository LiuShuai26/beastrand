"""Tests for StrandBus ZMQ wrapper."""

import os
import struct
import threading
import time

import pytest
import zmq

from beastrand.strandbus.strandbus import StrandBus, SBQueueFull, SBClosed


def _ep(suffix: str) -> str:
    """Generate a unique IPC endpoint for this test run."""
    return f"ipc:///tmp/beatstrand/test_{os.getpid()}_{suffix}"


class TestPushPull:

    def test_basic_send_recv(self):
        ep = _ep("basic")
        server = StrandBus()
        client = StrandBus()
        server.open("rx", "pull", ep, bind=True)
        client.open("tx", "push", ep, bind=False)

        client.send("tx", b"hello")
        msgs = server.recv_many("rx")
        assert msgs == [b"hello"]

        client.close_all()
        server.close_all()

    def test_multiple_messages(self):
        ep = _ep("multi")
        server = StrandBus()
        client = StrandBus()
        server.open("rx", "pull", ep, bind=True)
        client.open("tx", "push", ep, bind=False)

        for i in range(10):
            client.send("tx", struct.pack("<i", i))

        time.sleep(0.05)  # let messages arrive
        msgs = server.recv_many("rx")
        values = [struct.unpack("<i", m)[0] for m in msgs]
        assert values == list(range(10))

        client.close_all()
        server.close_all()

    def test_recv_many_blocks_for_first(self):
        """recv_many should block until at least one message arrives."""
        ep = _ep("block")
        server = StrandBus()
        client = StrandBus()
        server.open("rx", "pull", ep, bind=True)
        client.open("tx", "push", ep, bind=False)

        def delayed_send():
            time.sleep(0.1)
            client.send("tx", b"delayed")

        t = threading.Thread(target=delayed_send)
        t.start()

        msgs = server.recv_many("rx")
        assert msgs == [b"delayed"]

        t.join()
        client.close_all()
        server.close_all()

    def test_recv_at_least(self):
        ep = _ep("atleast")
        server = StrandBus()
        client = StrandBus()
        server.open("rx", "pull", ep, bind=True)
        client.open("tx", "push", ep, bind=False)

        for i in range(5):
            client.send("tx", struct.pack("<i", i))

        time.sleep(0.05)
        msgs = server.recv_at_least("rx", min_num=3)
        assert len(msgs) >= 3

        client.close_all()
        server.close_all()

    def test_recv_at_most(self):
        ep = _ep("atmost")
        server = StrandBus()
        client = StrandBus()
        server.open("rx", "pull", ep, bind=True)
        client.open("tx", "push", ep, bind=False)

        for i in range(10):
            client.send("tx", struct.pack("<i", i))

        time.sleep(0.05)
        msgs = server.recv_at_most("rx", max_num=3)
        assert len(msgs) <= 3

        client.close_all()
        server.close_all()


class TestPubSub:

    def test_basic_pubsub(self):
        ep = _ep("pubsub")
        pub = StrandBus()
        sub = StrandBus()
        pub.open("pub", "pub", ep, bind=True)
        sub.open("sub", "sub", ep, bind=False)

        time.sleep(0.1)  # ZMQ SUB needs time to establish subscription

        pub.send("pub", b"broadcast")
        time.sleep(0.05)

        ready = sub.poll(timeout_ms=500)
        assert "sub" in ready

        msg = sub.recv("sub")
        assert msg == b"broadcast"

        pub.close_all()
        sub.close_all()


class TestMultipart:

    def test_send_recv_multipart(self):
        ep = _ep("multipart")
        server = StrandBus()
        client = StrandBus()
        server.open("rx", "pull", ep, bind=True)
        client.open("tx", "push", ep, bind=False)

        client.send_multipart("tx", [b"frame1", b"frame2", b"frame3"])
        time.sleep(0.05)

        frames = server.recv_multipart("rx", noblock=False)
        assert frames == [b"frame1", b"frame2", b"frame3"]

        client.close_all()
        server.close_all()


class TestErrors:

    def test_duplicate_socket_name(self):
        ep = _ep("dup")
        bus = StrandBus()
        bus.open("sock", "push", ep, bind=True)
        with pytest.raises(ValueError, match="already open"):
            bus.open("sock", "push", _ep("dup2"), bind=True)
        bus.close_all()

    def test_invalid_mode(self):
        bus = StrandBus()
        with pytest.raises(ValueError, match="mode must be"):
            bus.open("sock", "req", _ep("inv"), bind=True)
        bus.close_all()

    def test_non_ipc_endpoint(self):
        bus = StrandBus()
        with pytest.raises(ValueError, match="ipc://"):
            bus.open("sock", "push", "tcp://127.0.0.1:5555", bind=True)
        bus.close_all()

    def test_recv_on_closed_socket(self):
        bus = StrandBus()
        with pytest.raises(SBClosed):
            bus.recv("nonexistent")

    def test_send_wrong_role(self):
        ep = _ep("wrongrole")
        bus = StrandBus()
        bus.open("rx", "pull", ep, bind=True)
        with pytest.raises(ValueError, match="expected"):
            bus.send("rx", b"data")
        bus.close_all()


class TestHWM:

    def test_set_hwm_does_not_crash(self):
        """Verify set_hwm is functional (actual drop behavior is ZMQ-internal)."""
        ep = _ep("hwm")
        bus = StrandBus()
        bus.open("tx", "push", ep, bind=True)
        bus.set_hwm("tx", snd=1)
        assert "tx" in bus.sockets
        bus.close_all()
