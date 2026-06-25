"""
Centralised API authentication.

This is the single place that knows *how to get an authenticated LLM client*.
The rest of the app (``translation``, ``summary``) just asks for a ready-to-use
client and never touches API keys directly.

Where the key comes from depends on where the code runs:

* **Locally** — from an environment variable (optionally via a ``.env`` file),
  e.g. ``GROQ_API_KEY`` / ``ANTHROPIC_API_KEY``.
* **On the Claude Code platform** — secrets are injected as a JSON document
  (``services.json``) rather than as environment variables, so we fall back to
  reading the key from there.

The environment is always checked first, so a local override wins over a file.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional


class AuthError(RuntimeError):
    """Raised when a required API key cannot be found anywhere."""


# Where the Claude Code platform drops injected service credentials. Overridable
# for tests or non-standard deployments.
_SERVICES_JSON = os.getenv("SERVICES_JSON_PATH", "services.json")


@lru_cache(maxsize=1)
def _load_services() -> dict:
    """Load and cache ``services.json`` if present; return ``{}`` otherwise."""
    try:
        with open(_SERVICES_JSON, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _key_from_services(*names: str) -> Optional[str]:
    """Look a key up in services.json, tolerating a few common layouts.

    Supports flat values (``{"GROQ_API_KEY": "..."}`` / ``{"groq": "..."}``)
    and nested provider objects (``{"groq": {"api_key": "..."}}``).
    """
    services = _load_services()
    for name in names:
        if name not in services:
            continue
        entry = services[name]
        if isinstance(entry, str) and entry.strip():
            return entry.strip()
        if isinstance(entry, dict):
            for field in ("api_key", "key", "value", "token", "secret"):
                value = entry.get(field)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def _resolve_key(env_var: str, *service_names: str) -> Optional[str]:
    """Return an API key from the environment, falling back to services.json."""
    value = os.getenv(env_var)
    if value and value.strip():
        return value.strip()
    return _key_from_services(env_var, *service_names)


def get_groq_client():
    """Return an authenticated Groq client (the default LLM provider)."""
    key = _resolve_key("GROQ_API_KEY", "groq", "Groq", "GROQ")
    if not key:
        raise AuthError(
            "No Groq API key found. Set GROQ_API_KEY in your environment (or a "
            ".env file), or provide it via services.json on the Claude Code "
            "platform. Get a key at https://console.groq.com/keys."
        )
    from groq import Groq  # imported lazily so transcription works without it

    return Groq(api_key=key)


def get_anthropic_client():
    """Return an authenticated Anthropic client (legacy / backwards-compat)."""
    key = _resolve_key("ANTHROPIC_API_KEY", "anthropic", "Anthropic")
    if not key:
        raise AuthError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY in your "
            "environment (or a .env file), or provide it via services.json."
        )
    import anthropic  # imported lazily

    return anthropic.Anthropic(api_key=key)
