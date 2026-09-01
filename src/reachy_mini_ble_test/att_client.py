"""Minimal GATT client over a raw LE L2CAP socket (the ATT channel).

Why not connect through BlueZ's D-Bus API? The robot's controller is
dual-mode and its advertisement does not carry the "BR/EDR Not Supported"
flag, so ``org.bluez.Device1.Connect`` picks the classic BR/EDR bearer —
where the robot has nothing listening — and fails with
``br-connection-profile-unavailable``. Phones and Web Bluetooth are LE-only
by construction, which is why they never hit this. This client does the
same thing they do: it opens an LE-bearer L2CAP connection on the ATT
channel (CID 4) directly, exactly like the classic ``gatttool``. It works
unprivileged, needs no host or robot configuration, and exercises the
robot's GATT service exactly as shipped.

Python 3.12's socket module cannot express the LE address type in a
BTPROTO_L2CAP sockaddr (3.13 added it), so bind/connect go through libc
with a hand-packed ``struct sockaddr_l2``.

Implements the small ATT subset the tests need: MTU exchange, service and
characteristic discovery, reads (with blob continuation), writes, CCCD
subscription, and notifications.
"""

import ctypes
import os
import queue
import socket
import struct
import time
from uuid import UUID

BDADDR_LE_PUBLIC = 1
BDADDR_LE_RANDOM = 2
ATT_CID = 4

# Linux-only socket constants, via getattr so type checking stays
# platform-independent (the suite itself only runs on Linux/BlueZ hosts).
_AF_BLUETOOTH: int = getattr(socket, "AF_BLUETOOTH", 31)
_BTPROTO_L2CAP: int = getattr(socket, "BTPROTO_L2CAP", 0)

# ATT opcodes
_ERROR_RSP = 0x01
_EXCHANGE_MTU_REQ, _EXCHANGE_MTU_RSP = 0x02, 0x03
_FIND_INFO_REQ, _FIND_INFO_RSP = 0x04, 0x05
_READ_BY_TYPE_REQ, _READ_BY_TYPE_RSP = 0x08, 0x09
_READ_REQ, _READ_RSP = 0x0A, 0x0B
_READ_BLOB_REQ, _READ_BLOB_RSP = 0x0C, 0x0D
_WRITE_REQ, _WRITE_RSP = 0x12, 0x13
_NOTIFICATION = 0x1B

_ATT_ERR_ATTRIBUTE_NOT_FOUND = 0x0A
_ATT_ERR_ATTRIBUTE_NOT_LONG = 0x0B

_CHARACTERISTIC = 0x2803
CCCD_UUID16 = 0x2902

_PROPERTY_BITS = {
    0x01: "broadcast",
    0x02: "read",
    0x04: "write-without-response",
    0x08: "write",
    0x10: "notify",
    0x20: "indicate",
}


def _uuid128_to_str(le_bytes: bytes) -> str:
    return str(UUID(bytes=bytes(reversed(le_bytes))))


def _uuid16_to_str(value: int) -> str:
    return f"{value:08x}-0000-1000-8000-00805f9b34fb"


def _uuid_bytes_to_str(uuid_bytes: bytes) -> str:
    if len(uuid_bytes) == 2:
        return _uuid16_to_str(struct.unpack("<H", uuid_bytes)[0])
    return _uuid128_to_str(uuid_bytes)


def _bdaddr(mac: str) -> bytes:
    return bytes(reversed(bytes.fromhex(mac.replace(":", ""))))


def _sockaddr_l2(mac_le: bytes, addr_type: int) -> bytes:
    # struct sockaddr_l2: family, psm, bdaddr[6], cid, bdaddr_type (+pad)
    return struct.pack("<HH6sHBx", _AF_BLUETOOTH, 0, mac_le, ATT_CID, addr_type)


class AttError(Exception):
    """ATT Error Response from the server."""

    def __init__(self, request_opcode: int, handle: int, code: int) -> None:
        """Record the failed request opcode, target handle, and error code."""
        self.request_opcode = request_opcode
        self.handle = handle
        self.code = code
        super().__init__(
            f"ATT error 0x{code:02x} for request 0x{request_opcode:02x} "
            f"on handle 0x{handle:04x}"
        )


class Characteristic:
    """A discovered characteristic: handles, UUID, decoded properties."""

    def __init__(
        self, decl_handle: int, properties_mask: int, value_handle: int, uuid: str
    ) -> None:
        """Store the discovery result for one characteristic declaration."""
        self.decl_handle = decl_handle
        self.value_handle = value_handle
        self.uuid = uuid
        self.properties = [
            name for bit, name in _PROPERTY_BITS.items() if properties_mask & bit
        ]


class AttClient:
    """Synchronous GATT client over a raw LE L2CAP (ATT) socket."""

    def __init__(
        self, address: str, address_type: str = "public", timeout: float = 10.0
    ) -> None:
        """Prepare a client for the peer at ``address`` (no I/O yet)."""
        self.address = address
        self.address_type = address_type
        self.timeout = timeout
        self.mtu = 23
        self.characteristics: list[Characteristic] = []  # in handle order
        self.notifications: queue.Queue[tuple[int, bytes]] = queue.Queue()
        self._sock: socket.socket | None = None

    # -- connection ----------------------------------------------------

    def connect(self) -> None:
        """Open the LE connection and discover the GATT database."""
        peer_type = (
            BDADDR_LE_RANDOM if self.address_type == "random" else BDADDR_LE_PUBLIC
        )
        libc = ctypes.CDLL(None, use_errno=True)
        sock = socket.socket(_AF_BLUETOOTH, socket.SOCK_SEQPACKET, _BTPROTO_L2CAP)
        try:
            local = _sockaddr_l2(b"\x00" * 6, BDADDR_LE_PUBLIC)
            if libc.bind(sock.fileno(), local, len(local)) != 0:
                raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
            peer = _sockaddr_l2(_bdaddr(self.address), peer_type)
            if libc.connect(sock.fileno(), peer, len(peer)) != 0:
                raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        except BaseException:
            sock.close()
            raise
        sock.settimeout(self.timeout)
        self._sock = sock
        self._exchange_mtu()
        self._discover()

    def close(self) -> None:
        """Close the LE connection."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    # -- ATT plumbing --------------------------------------------------

    def _socket(self) -> socket.socket:
        if self._sock is None:
            raise ConnectionError("not connected")
        return self._sock

    def _recv(self, deadline: float) -> bytes:
        sock = self._socket()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("timed out waiting for ATT PDU")
        sock.settimeout(remaining)
        pdu = sock.recv(1024)
        if not pdu:
            raise ConnectionError("LE connection closed by peer")
        return pdu

    def _request(self, pdu: bytes, expected_opcode: int) -> bytes:
        """Send one ATT request and return the matching response PDU.

        Notifications interleaved with the response are queued, never lost.
        """
        self._socket().send(pdu)
        deadline = time.monotonic() + self.timeout
        while True:
            rsp = self._recv(deadline)
            opcode = rsp[0]
            if opcode == _NOTIFICATION:
                handle = int(struct.unpack_from("<H", rsp, 1)[0])
                self.notifications.put((handle, rsp[3:]))
            elif opcode == expected_opcode:
                return rsp
            elif opcode == _ERROR_RSP and rsp[1] == pdu[0]:
                _, req_op, handle, code = struct.unpack("<BBHB", rsp)
                raise AttError(req_op, handle, code)
            # anything else (e.g. server-initiated exchange): ignore

    def wait_notification(self, timeout: float) -> bytes:
        """Return the next notification payload (bytes) within timeout.

        Drains PDUs from the socket into the queue as needed.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                return self.notifications.get_nowait()[1]
            except queue.Empty:
                pass
            rsp = self._recv(deadline)
            if rsp[0] == _NOTIFICATION:
                handle = int(struct.unpack_from("<H", rsp, 1)[0])
                self.notifications.put((handle, rsp[3:]))

    def drain_notifications(self) -> None:
        """Discard any queued notifications (before sending a new command)."""
        while True:
            try:
                self.notifications.get_nowait()
            except queue.Empty:
                return

    # -- GATT operations -----------------------------------------------

    def _exchange_mtu(self, mtu: int = 517) -> None:
        rsp = self._request(
            struct.pack("<BH", _EXCHANGE_MTU_REQ, mtu), _EXCHANGE_MTU_RSP
        )
        self.mtu = min(mtu, int(struct.unpack_from("<H", rsp, 1)[0]))

    def _discover(self) -> None:
        # Characteristics: Read By Type on the declaration UUID.
        start = decl_handle = 0x0001
        while start <= 0xFFFF:
            try:
                rsp = self._request(
                    struct.pack(
                        "<BHHH", _READ_BY_TYPE_REQ, start, 0xFFFF, _CHARACTERISTIC
                    ),
                    _READ_BY_TYPE_RSP,
                )
            except AttError as e:
                if e.code == _ATT_ERR_ATTRIBUTE_NOT_FOUND:
                    break
                raise
            entry_len, offset = rsp[1], 2
            while offset + entry_len <= len(rsp):
                decl_handle = int(struct.unpack_from("<H", rsp, offset)[0])
                props, value_handle = struct.unpack_from("<BH", rsp, offset + 2)
                uuid = _uuid_bytes_to_str(rsp[offset + 5 : offset + entry_len])
                self.characteristics.append(
                    Characteristic(decl_handle, int(props), int(value_handle), uuid)
                )
                offset += entry_len
            start = decl_handle + 1

    def characteristic(self, uuid: str) -> Characteristic | None:
        """Return the first discovered characteristic with this UUID, or None."""
        uuid = uuid.lower()
        for char in self.characteristics:
            if char.uuid == uuid:
                return char
        return None

    def _descriptor_handle(self, char: Characteristic, uuid16: int) -> int | None:
        """Find a descriptor of ``char`` by 16-bit UUID via Find Information."""
        later = [
            c.decl_handle
            for c in self.characteristics
            if c.decl_handle > char.decl_handle
        ]
        end = min(later) - 1 if later else 0xFFFF
        start = handle = char.value_handle + 1
        while start <= end:
            try:
                rsp = self._request(
                    struct.pack("<BHH", _FIND_INFO_REQ, start, end), _FIND_INFO_RSP
                )
            except AttError as e:
                if e.code == _ATT_ERR_ATTRIBUTE_NOT_FOUND:
                    return None
                raise
            fmt, offset = rsp[1], 2
            entry_len = 4 if fmt == 1 else 18
            while offset + entry_len <= len(rsp):
                handle = int(struct.unpack_from("<H", rsp, offset)[0])
                if fmt == 1:
                    if struct.unpack_from("<H", rsp, offset + 2)[0] == uuid16:
                        return handle
                offset += entry_len
            start = handle + 1
        return None

    def read(self, handle: int) -> bytes:
        """Read an attribute value, continuing with Read Blob when long."""
        rsp = self._request(struct.pack("<BH", _READ_REQ, handle), _READ_RSP)
        value = rsp[1:]
        while len(value) % (self.mtu - 1) == 0 and value:
            try:
                rsp = self._request(
                    struct.pack("<BHH", _READ_BLOB_REQ, handle, len(value)),
                    _READ_BLOB_RSP,
                )
            except AttError as e:
                if e.code == _ATT_ERR_ATTRIBUTE_NOT_LONG:
                    break  # value was exactly a multiple of MTU-1
                raise
            if len(rsp) <= 1:
                break
            value += rsp[1:]
        return bytes(value)

    def write(self, handle: int, data: bytes) -> None:
        """Write an attribute value (Write Request, acknowledged)."""
        self._request(struct.pack("<BH", _WRITE_REQ, handle) + data, _WRITE_RSP)

    def subscribe(self, char: Characteristic) -> None:
        """Enable notifications on a characteristic via its CCCD."""
        cccd = self._descriptor_handle(char, CCCD_UUID16)
        if cccd is None:
            raise RuntimeError(f"no CCCD found for characteristic {char.uuid}")
        self.write(cccd, b"\x01\x00")
