"""LLM client factory + resilient structured-output calls.

Supports two OpenAI-compatible providers and auto-detects which one your key
belongs to from its prefix:

    xai-...  ->  xAI Grok   (paid)               https://console.x.ai
    gsk_...  ->  Groq       (free tier available) https://console.groq.com/keys

NOTE: "Grok" (xAI) and "Groq" (groq.com) are different companies.

Temporary errors (503 overloaded, 504 deadline, 429 rate limit) are retried with
exponential backoff; if a model keeps failing, the next model in
LLM_FALLBACK_MODELS (comma-separated) is tried.
"""

import os
import sys
import time
from functools import lru_cache
from importlib import import_module

try:
    load_dotenv = import_module("dotenv").load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        """Allow environment-based configuration when python-dotenv is absent."""
        return False

PROVIDERS = {
    "xai": {
        "label": "xAI Grok",
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-4.3",
        "key_prefix": "xai-",
    },
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "openai/gpt-oss-20b",
        "key_prefix": "gsk_",
    },
}
# Env vars checked for a key, in order. Any of them may hold either kind of key.
KEY_ENV_VARS = ("XAI_API_KEY", "GROK_API_KEY", "GROQ_API_KEY")

RETRIES_PER_MODEL = 4
BACKOFF_SECONDS = 2.0  # waits 2s, 4s, 8s between attempts
REQUEST_TIMEOUT = 120  # seconds per request

_BAD_KEY = ("INCORRECT API KEY", "INVALID API KEY", "INVALID_API_KEY", "UNAUTHORIZED",
            "ERROR CODE: 401")
_TRANSIENT = (
    "503", "504", "429", "UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED",
    "TIMED OUT", "TIMEOUT", "OVERLOADED", "HIGH DEMAND", "RATE LIMIT", "CONNECTION ERROR",
    "TOOL_USE_FAILED", "FAILED TO CALL A FUNCTION",  # Groq/Llama occasionally mangles a tool call
)
_MODEL_GONE = ("404", "NOT_FOUND", "NO LONGER AVAILABLE", "DOES NOT EXIST")


class ConfigError(RuntimeError):
    """Raised for configuration problems (missing or rejected API key)."""


def _find_key():
    """Return (api_key, env_var_name) for the first key found, else (None, None)."""
    load_dotenv()
    for name in KEY_ENV_VARS:
        value = (os.getenv(name) or "").strip().strip("\"'")
        if value:
            return value, name
    return None, None


def _detect_provider(key, env_name) -> str:
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit in PROVIDERS:
        return explicit
    if key:
        for name, cfg in PROVIDERS.items():
            if key.startswith(cfg["key_prefix"]):
                return name
    return "groq" if env_name == "GROQ_API_KEY" else "xai"


def resolve_provider():
    """Return (provider_name, api_key). Raises ConfigError if no key is set."""
    key, env_name = _find_key()
    if not key:
        raise ConfigError(
            "No API key found. Put one in your .env file (copy .env.example to .env):\n"
            "  GROQ_API_KEY=gsk_...   (Groq, free tier: https://console.groq.com/keys)\n"
            "  XAI_API_KEY=xai-...    (xAI Grok, paid: https://console.x.ai)"
        )
    return _detect_provider(key, env_name), key


def has_api_key() -> bool:
    return _find_key()[0] is not None


def model_chain() -> list:
    key, env_name = _find_key()
    provider = _detect_provider(key, env_name)
    primary = (
        os.getenv("LLM_MODEL") or os.getenv("GROK_MODEL") or PROVIDERS[provider]["default_model"]
    )
    raw = os.getenv("LLM_FALLBACK_MODELS") or os.getenv("GROK_FALLBACK_MODELS") or ""
    chain = []
    for m in [primary, *[x.strip() for x in raw.split(",")]]:
        if m and m not in chain:
            chain.append(m)
    return chain


@lru_cache(maxsize=None)
def get_llm(model: str = None):
    provider, key = resolve_provider()
    try:
        ChatOpenAI = import_module("langchain_openai").ChatOpenAI
    except ModuleNotFoundError as exc:
        if exc.name == "langchain_openai":
            raise ConfigError(
                "Missing dependency 'langchain-openai'. Install it with: "
                "pip install langchain-openai"
            ) from exc
        raise
    return ChatOpenAI(
        model=model or model_chain()[0],
        api_key=key,
        base_url=PROVIDERS[provider]["base_url"],
        temperature=0,
        max_retries=1,  # our own loop below handles the real retrying
        timeout=REQUEST_TIMEOUT,
    )


def _classify(exc: Exception) -> str:
    text = str(exc).upper()
    if any(m in text for m in _BAD_KEY):
        return "bad_key"
    if any(m in text for m in _MODEL_GONE):
        return "model_gone"
    if any(m in text for m in _TRANSIENT):
        return "transient"
    return "fatal"


def _structured_output_method(model: str) -> str:
    if model.startswith("openai/gpt-oss-"):
        return "json_schema"
    return "function_calling"


def _bad_key_error(exc: Exception) -> ConfigError:
    key, env_name = _find_key()
    provider = _detect_provider(key, env_name)
    shown = f"'{key[:4]}…' ({len(key)} chars, from {env_name})" if key else "none"
    return ConfigError(
        f"{PROVIDERS[provider]['label']} rejected your API key. Key I sent: {shown}.\n"
        "xAI Grok keys start with 'xai-' (paid, console.x.ai); "
        "Groq keys start with 'gsk_' (console.groq.com/keys).\n"
        "Create a fresh key, paste it into .env with no quotes or spaces, and restart."
    )


def invoke_structured(schema, prompt: str):
    """Call the LLM and return a `schema` instance, retrying/falling back as needed."""
    last_error = None
    for model in model_chain():
        for attempt in range(RETRIES_PER_MODEL):
            try:
                llm = get_llm(model).with_structured_output(
                    schema, method=_structured_output_method(model)
                )
                return llm.invoke(prompt)
            except ConfigError:
                raise
            except Exception as exc:
                kind = _classify(exc)
                if kind == "bad_key":
                    raise _bad_key_error(exc) from exc
                if kind == "fatal":
                    raise
                last_error = exc
                if kind == "model_gone":
                    print(f"[{model}] unavailable, trying next model...", file=sys.stderr)
                    break
                if attempt < RETRIES_PER_MODEL - 1:
                    wait = BACKOFF_SECONDS * (2 ** attempt)
                    print(
                        f"[{model}] temporary error, retrying in {wait:.0f}s "
                        f"({attempt + 1}/{RETRIES_PER_MODEL - 1})...",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
    raise last_error
