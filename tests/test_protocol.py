import socket

from edgeavi.ipc.protocol import recv_message, send_message


def test_roundtrip():
    a, b = socket.socketpair()
    send_message(a, {"type": "frame", "product_id": "P1"}, b"\x00\x01" * 5000)
    header, payload = recv_message(b)
    assert header["product_id"] == "P1" and len(payload) == 10000
    send_message(a, {"type": "shutdown"})
    assert recv_message(b) == ({"type": "shutdown"}, b"")
