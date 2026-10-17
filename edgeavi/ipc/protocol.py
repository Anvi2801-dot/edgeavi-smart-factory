"""Tiny length-prefixed TCP protocol between the Webots controller and the engine.

Wire format:  [u32 header_len][u32 payload_len][header JSON][payload bytes]
Message types (header["type"]): frame, result, shutdown, error
"""
from __future__ import annotations

import json
import socket
import struct

_PREFIX = struct.Struct("!II")
MAX_PAYLOAD = 64 * 1024 * 1024


def send_message(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    h = json.dumps(header).encode("utf-8")
    sock.sendall(_PREFIX.pack(len(h), len(payload)) + h + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def recv_message(sock: socket.socket) -> tuple[dict, bytes]:
    hlen, plen = _PREFIX.unpack(_recv_exact(sock, _PREFIX.size))
    if plen > MAX_PAYLOAD:
        raise ValueError(f"payload too large: {plen}")
    header = json.loads(_recv_exact(sock, hlen).decode("utf-8"))
    payload = _recv_exact(sock, plen) if plen else b""
    return header, payload


def tune(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
