"""Test helpers: a fake Grok that returns queued structured responses."""

import pytest

from task_organizer import llm as llm_module


class _Runnable:
    def __init__(self, queue):
        self.queue = queue
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        item = self.queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeLLM:
    """responses: {SchemaClass: [response_or_exception, ...]} consumed in order."""

    def __init__(self, responses):
        self.responses = responses
        self.runnables = {}

    def with_structured_output(self, schema, **kwargs):
        if schema not in self.runnables:
            self.runnables[schema] = _Runnable(self.responses[schema])
        return self.runnables[schema]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(llm_module.time, "sleep", lambda s: None)


@pytest.fixture
def use_fake_llm(monkeypatch):
    def _install(responses):
        fake = FakeLLM(responses)
        monkeypatch.setattr(llm_module, "get_llm", lambda model=None: fake)
        return fake

    return _install
