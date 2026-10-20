import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager

from task_organizer import llm as llm_module
from task_organizer.llm import ConfigError
from task_organizer.schemas import (
    ExtractedTask,
    ExtractedTasks,
    PrioritizedTasks,
    PriorityItem,
)
from task_organizer.web import make_server


@contextmanager
def running_server():
    server = make_server("127.0.0.1", 0)  # port 0 = pick a free port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _post(base, payload, raw=None):
    body = raw if raw is not None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + "/api/organize", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_serves_the_page():
    with running_server() as base:
        with urllib.request.urlopen(base + "/") as r:
            html = r.read().decode("utf-8")
            assert r.status == 200
    assert "to-do bestie" in html and "/api/organize" in html


def test_organize_endpoint_returns_sorted_tasks(use_fake_llm):
    extracted = ExtractedTasks(
        tasks=[
            ExtractedTask(title="Buy groceries", deadline="today"),
            ExtractedTask(title="Finish assignment", deadline="tomorrow"),
        ]
    )
    prio = PrioritizedTasks(
        items=[
            PriorityItem(id=1, priority="Medium", reason="errand"),
            PriorityItem(id=2, priority="High", reason="due soon"),
        ]
    )
    use_fake_llm({ExtractedTasks: [extracted], PrioritizedTasks: [prio]})
    with running_server() as base:
        status, data = _post(base, {"text": "groceries and assignment"})
    assert status == 200
    assert [t["title"] for t in data["tasks"]] == ["Finish assignment", "Buy groceries"]


def test_empty_text_is_400():
    with running_server() as base:
        status, data = _post(base, {"text": "   "})
    assert status == 400 and "error" in data


def test_bad_json_is_400():
    with running_server() as base:
        status, data = _post(base, None, raw=b"not json{")
    assert status == 400


def test_unknown_route_is_404():
    with running_server() as base:
        try:
            urllib.request.urlopen(base + "/nope")
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404


def test_missing_api_key_is_reported(monkeypatch):
    def boom(model=None):
        raise ConfigError("No API key found.")

    monkeypatch.setattr(llm_module, "get_llm", boom)
    with running_server() as base:
        status, data = _post(base, {"text": "do stuff"})
    assert status == 500 and "API key" in data["error"]


def test_upstream_failure_is_502(monkeypatch):
    class Broken:
        def with_structured_output(self, schema, **kw):
            return self

        def invoke(self, prompt):
            raise ValueError("400 bad request")  # non-retryable

    monkeypatch.setattr(llm_module, "get_llm", lambda model=None: Broken())
    with running_server() as base:
        status, data = _post(base, {"text": "do stuff"})
    assert status == 502 and "AI" in data["error"]
