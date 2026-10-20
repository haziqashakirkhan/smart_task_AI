import pytest

from task_organizer import llm as llm_module
from task_organizer.llm import ConfigError, PROVIDERS


def _clean_env(monkeypatch):
    monkeypatch.setattr(llm_module, "load_dotenv", lambda *a, **k: None)
    for name in ("XAI_API_KEY", "GROK_API_KEY", "GROQ_API_KEY", "LLM_PROVIDER",
                 "LLM_MODEL", "GROK_MODEL", "LLM_FALLBACK_MODELS", "GROK_FALLBACK_MODELS"):
        monkeypatch.delenv(name, raising=False)


def test_gsk_key_in_xai_var_is_detected_as_groq(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "gsk_abc123")
    assert llm_module.resolve_provider() == ("groq", "gsk_abc123")


def test_xai_prefix_is_detected_as_xai(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "xai-abc123")
    assert llm_module.resolve_provider() == ("xai", "xai-abc123")


def test_quotes_and_spaces_are_stripped(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", ' "gsk_abc123" ')
    assert llm_module.resolve_provider() == ("groq", "gsk_abc123")


def test_explicit_provider_wins(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "weird-key")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert llm_module.resolve_provider()[0] == "groq"


def test_missing_key_raises_config_error(monkeypatch):
    _clean_env(monkeypatch)
    with pytest.raises(ConfigError):
        llm_module.resolve_provider()
    assert llm_module.has_api_key() is False


def test_default_model_follows_provider(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_abc")
    assert PROVIDERS["groq"]["default_model"] == "openai/gpt-oss-20b"
    assert llm_module.model_chain()[0] == "openai/gpt-oss-20b"
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.setenv("XAI_API_KEY", "xai-abc")
    assert llm_module.model_chain()[0] == PROVIDERS["xai"]["default_model"]


def test_gpt_oss_uses_json_schema_output(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_abc")
    methods = []

    class Structured:
        def invoke(self, prompt):
            return "ok"

    class Model:
        def with_structured_output(self, schema, **kwargs):
            methods.append(kwargs["method"])
            return Structured()

    monkeypatch.setattr(llm_module, "get_llm", lambda model=None: Model())
    assert llm_module.invoke_structured(object, "p") == "ok"
    assert methods == ["json_schema"]


def test_other_models_keep_function_calling(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_abc")
    methods = []

    class Structured:
        def invoke(self, prompt):
            return "ok"

    class Model:
        def with_structured_output(self, schema, **kwargs):
            methods.append(kwargs["method"])
            return Structured()

    monkeypatch.setenv("LLM_MODEL", "other-model")
    monkeypatch.setattr(llm_module, "get_llm", lambda model=None: Model())
    assert llm_module.invoke_structured(object, "p") == "ok"
    assert methods == ["function_calling"]


def test_rejected_key_becomes_helpful_config_error(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("XAI_API_KEY", "gsk_wrong")

    class Rejecting:
        def with_structured_output(self, schema, **kw):
            return self

        def invoke(self, prompt):
            raise RuntimeError("Error code: 400 - Incorrect API key provided.")

    monkeypatch.setattr(llm_module, "get_llm", lambda model=None: Rejecting())
    with pytest.raises(ConfigError) as info:
        llm_module.invoke_structured(object, "p")
    message = str(info.value)
    assert "gsk_" in message and "wrong" not in message  # shows prefix, never the full key


def test_tool_use_failed_is_retried(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_abc")
    calls = []

    class Flaky:
        def with_structured_output(self, schema, **kw):
            return self

        def invoke(self, prompt):
            calls.append(1)
            if len(calls) < 2:
                raise RuntimeError("400 tool_use_failed: Failed to call a function")
            return "ok"

    monkeypatch.setattr(llm_module, "get_llm", lambda model=None: Flaky())
    assert llm_module.invoke_structured(object, "p") == "ok"
    assert len(calls) == 2
