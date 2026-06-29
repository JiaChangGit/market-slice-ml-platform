#!/usr/bin/env python3
"""Detect the Windows host address used by historical IBKR probes from WSL2."""

from __future__ import annotations

import platform
import socket
import subprocess
from pathlib import Path


def is_wsl2() -> bool:
    release = platform.uname().release.lower()
    return "microsoft" in release or "wsl" in release


def get_wsl2_host_ip() -> str | None:
    resolv = Path("/etc/resolv.conf")
    if resolv.exists():
        for line in resolv.read_text(encoding="utf-8").splitlines():
            if line.startswith("nameserver"):
                candidate = line.split()[-1]
                if candidate and not candidate.startswith("127."):
                    return candidate
    try:
        output = subprocess.check_output(["ip", "route", "show", "default"], text=True)
        parts = output.split()
        if "via" in parts:
            return parts[parts.index("via") + 1]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        return None
    return None


def resolve_ibkr_host(configured_host: str = "auto") -> str:
    if configured_host != "auto":
        return configured_host
    return get_wsl2_host_ip() if is_wsl2() and get_wsl2_host_ip() else "127.0.0.1"


def probe_socket(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> None:
    host = resolve_ibkr_host()
    print(f"WSL2: {is_wsl2()}")
    print(f"Detected IBKR host: {host}")
    for port in (7497, 4002):
        state = "reachable" if probe_socket(host, port) else "closed"
        print(f"Historical API socket {host}:{port}: {state}")


if __name__ == "__main__":
    main()
