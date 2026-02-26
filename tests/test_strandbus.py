#!/usr/bin/env python3
import multiprocessing as mp
import os
import time
import zmq
from strandbus.strandbus import StrandBus

_BASE = f"ipc:///tmp/beatstrand/test_{os.getpid()}"


def server():
    bus = StrandBus()
    bus.open("pull", "pull", f"{_BASE}/test.req", bind=True)
    print("[server] waiting for messages...")
    while True:
        try:
            msg = bus.recv("pull", noblock=True)
            print("[server] got:", msg.decode())
            if msg == b"quit":
                break
        except zmq.Again:
            time.sleep(0.05)  # nothing yet
    bus.close_all()
    print("[server] stopped")


def client():
    time.sleep(0.5)  # wait for server bind
    bus = StrandBus()
    bus.open("push", "push", f"{_BASE}/test.req", bind=False)

    for i in range(5):
        msg = f"hello {i}".encode()
        bus.send("push", msg)
        print("[client] sent:", msg.decode())
        time.sleep(0.2)

    bus.send("push", b"quit")
    bus.close_all()
    print("[client] stopped")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    p_server = mp.Process(target=server)
    p_client = mp.Process(target=client)

    p_server.start()
    p_client.start()

    p_client.join()
    p_server.join()
    print("test done")
