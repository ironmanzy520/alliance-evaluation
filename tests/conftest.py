"""Shared boundaries for offline engineering tests."""

import socket

import pytest


@pytest.fixture(autouse=True)
def no_network_or_credentials(monkeypatch):
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "MISTRAL_API_KEY", "ANTHROPIC_API_KEY", "ALLIANCE_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def reject(*args, **kwargs):
        raise AssertionError("Offline engineering tests must not connect to a network")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)
    monkeypatch.setattr(socket, "create_connection", reject)
