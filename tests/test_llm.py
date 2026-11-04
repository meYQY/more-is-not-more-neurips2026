"""Tests for llm.py — client creation and config handling."""

import os

import pytest

from modbench.llm import _get_client, _clients


@pytest.fixture(autouse=True)
def clear_client_cache():
    """Clear client cache between tests."""
    _clients.clear()
    yield
    _clients.clear()


def test_get_client_missing_env_var():
    """Should raise ValueError with helpful message when env var is missing."""
    config = {
        "model": {
            "name": "test",
            "base_url": "https://example.com",
            "api_key_env": "NONEXISTENT_KEY_FOR_TEST",
        }
    }
    os.environ.pop("NONEXISTENT_KEY_FOR_TEST", None)
    with pytest.raises(ValueError, match="NONEXISTENT_KEY_FOR_TEST"):
        _get_client(config)


def test_get_client_caches_by_config(monkeypatch: pytest.MonkeyPatch):
    """Same (base_url, api_key_env) should return the same client."""
    monkeypatch.setenv("TEST_KEY_A", "key-a")
    config = {
        "model": {
            "name": "test",
            "base_url": "https://example.com",
            "api_key_env": "TEST_KEY_A",
        }
    }
    client1 = _get_client(config)
    client2 = _get_client(config)
    assert client1 is client2


def test_get_client_different_providers(monkeypatch: pytest.MonkeyPatch):
    """Different (base_url, api_key_env) should return different clients."""
    monkeypatch.setenv("TEST_KEY_A", "key-a")
    monkeypatch.setenv("TEST_KEY_B", "key-b")
    config_a = {
        "model": {
            "name": "model-a",
            "base_url": "https://a.com",
            "api_key_env": "TEST_KEY_A",
        }
    }
    config_b = {
        "model": {
            "name": "model-b",
            "base_url": "https://b.com",
            "api_key_env": "TEST_KEY_B",
        }
    }
    client_a = _get_client(config_a)
    client_b = _get_client(config_b)
    assert client_a is not client_b
    assert len(_clients) == 2
