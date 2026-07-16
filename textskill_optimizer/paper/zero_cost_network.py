"""Process-wide Python network guard used only by the M6 gate."""

from __future__ import annotations

import socket


_MESSAGE = "M6 zero-cost gate blocks external network"
_NETWORK_FAMILIES = {socket.AF_INET, socket.AF_INET6}
_ORIGINAL_SOCKET = socket.socket
_installed = False


class _GuardedSocket(_ORIGINAL_SOCKET):
    def connect(self, address: object) -> None:
        if self.family in _NETWORK_FAMILIES:
            raise OSError(_MESSAGE)
        return super().connect(address)

    def connect_ex(self, address: object) -> int:
        if self.family in _NETWORK_FAMILIES:
            raise OSError(_MESSAGE)
        return super().connect_ex(address)


def install_zero_cost_network_guard() -> None:
    """Block DNS and IPv4/IPv6 connects for this process and its Python code."""

    global _installed
    if _installed:
        return

    def blocked(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise OSError(_MESSAGE)

    socket.socket = _GuardedSocket
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    socket.gethostbyname = blocked
    socket.gethostbyname_ex = blocked
    _installed = True


def zero_cost_network_guard_active() -> bool:
    """Return whether this process installed the M6 network guard."""

    return _installed
