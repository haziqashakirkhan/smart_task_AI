import json

from task_organizer import organize_tasks
from task_organizer.cli import main
from task_organizer.formatters import render_json, render_markdown, render_text
from task_organizer.schemas import (
    ExtractedTask,
    ExtractedTasks,
    PrioritizedTasks,
    PriorityItem,
)

EXTRACTED = ExtractedTasks(
    tasks=[
        ExtractedTask(title="Finish assignment", deadline="tomorrow"),
        ExtractedTask(title="Buy groceries", deadline="today"),
        ExtractedTask(title="Prepare for meeting", deadline="next week"),
    ]
)

FULL_PRIORITIES = PrioritizedTasks(
    items=[
        PriorityItem(id=1, priority="High", reason="Due tomorrow."),
        PriorityItem(id=2, priority="Medium", reason="Everyday errand."),
        PriorityItem(id=3, priority="Low", reason="A week away."),
    ]
)


def test_happy_path(use_fake_llm):
    use_fake_llm({ExtractedTasks: [EXTRACTED], PrioritizedTasks: [FULL_PRIORITIES]})
    state = organize_tasks("anything")
    final = state["final_tasks"]
    assert [t["title"] for t in final] == [
        "Finish assignment",
        "Buy groceries",
        "Prepare for meeting",
    ]
    assert [t["priority"] for t in final] == ["High", "Medium", "Low"]
    assert "HIGH PRIORITY" in state["formatted_output"]


def test_output_sorted_by_priority(use_fake_llm):
    shuffled = PrioritizedTasks(
        items=[
            PriorityItem(id=1, priority="Low", reason="r"),
            PriorityItem(id=2, priority="High", reason="r"),
            PriorityItem(id=3, priority="Medium", reason="r"),
        ]
    )
    use_fake_llm({ExtractedTasks: [EXTRACTED], PrioritizedTasks: [shuffled]})
    final = organize_tasks("x")["final_tasks"]
    assert [t["priority"] for t in final] == ["High", "Medium", "Low"]


def test_retry_fills_missing_tasks(use_fake_llm):
    first = PrioritizedTasks(items=[PriorityItem(id=1, priority="High", reason="r")])
    second = PrioritizedTasks(
        items=[
            PriorityItem(id=2, priority="Medium", reason="r"),
            PriorityItem(id=3, priority="Low", reason="r"),
        ]
    )
    fake = use_fake_llm({ExtractedTasks: [EXTRACTED], PrioritizedTasks: [first, second]})
    final = organize_tasks("x")["final_tasks"]
    assert {t["id"]: t["priority"] for t in final} == {1: "High", 2: "Medium", 3: "Low"}
    # The retry prompt should only mention the tasks that were still missing.
    retry_prompt = fake.runnables[PrioritizedTasks].prompts[1]
    assert "[1]" not in retry_prompt and "[2]" in retry_prompt and "[3]" in retry_prompt


def test_defaults_to_medium_after_max_attempts(use_fake_llm):
    empty = PrioritizedTasks(items=[])
    use_fake_llm({ExtractedTasks: [EXTRACTED], PrioritizedTasks: [empty, empty, empty]})
    final = organize_tasks("x")["final_tasks"]
    assert len(final) == 3
    assert all(t["priority"] == "Medium" for t in final)


def test_ignores_hallucinated_and_duplicate_ids(use_fake_llm):
    noisy = PrioritizedTasks(
        items=[
            PriorityItem(id=99, priority="High", reason="ghost"),
            PriorityItem(id=1, priority="High", reason="real"),
            PriorityItem(id=1, priority="Low", reason="duplicate"),
            PriorityItem(id=2, priority="Low", reason="r"),
            PriorityItem(id=3, priority="Low", reason="r"),
        ]
    )
    use_fake_llm({ExtractedTasks: [EXTRACTED], PrioritizedTasks: [noisy]})
    final = organize_tasks("x")["final_tasks"]
    assert len(final) == 3
    assert next(t for t in final if t["id"] == 1)["reason"] == "real"


def test_no_tasks_skips_prioritization(use_fake_llm):
    fake = use_fake_llm({ExtractedTasks: [ExtractedTasks(tasks=[])], PrioritizedTasks: []})
    state = organize_tasks("hello there")
    assert state["final_tasks"] == []
    assert "No tasks found" in state["formatted_output"]
    assert PrioritizedTasks not in fake.runnables  # never called


def test_null_like_deadlines_are_cleaned(use_fake_llm):
    extracted = ExtractedTasks(tasks=[ExtractedTask(title="Call mom", deadline="null")])
    prio = PrioritizedTasks(items=[PriorityItem(id=1, priority="Low", reason="r")])
    use_fake_llm({ExtractedTasks: [extracted], PrioritizedTasks: [prio]})
    assert organize_tasks("x")["final_tasks"][0]["deadline"] is None


def test_empty_input_rejected():
    import pytest

    with pytest.raises(ValueError):
        organize_tasks("   ")


def test_formatters():
    tasks = [
        {"id": 1, "title": "A", "deadline": "today", "priority": "High", "reason": "r1"},
        {"id": 2, "title": "B", "deadline": None, "priority": "Low", "reason": "r2"},
    ]
    assert "(due: today)" in render_text(tasks)
    assert "- [ ] **A**" in render_markdown(tasks)
    assert json.loads(render_json(tasks))[1]["title"] == "B"


def test_cli_json_output(use_fake_llm, tmp_path, capsys):
    use_fake_llm({ExtractedTasks: [EXTRACTED], PrioritizedTasks: [FULL_PRIORITIES]})
    src = tmp_path / "tasks.txt"
    src.write_text("finish assignment, groceries, meeting prep", encoding="utf-8")
    out = tmp_path / "out.json"

    code = main(["--file", str(src), "--format", "json", "-o", str(out)])

    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed[0]["priority"] == "High"
    assert json.loads(out.read_text(encoding="utf-8")) == printed


# ---------------------------------------------------------------------------
# Resilience: retries and model fallback
# ---------------------------------------------------------------------------
def test_retries_transient_errors_then_succeeds(use_fake_llm):
    use_fake_llm(
        {
            ExtractedTasks: [
                RuntimeError("503 UNAVAILABLE high demand"),
                RuntimeError("504 DEADLINE_EXCEEDED"),
                EXTRACTED,
            ],
            PrioritizedTasks: [FULL_PRIORITIES],
        }
    )
    assert len(organize_tasks("x")["final_tasks"]) == 3


def _model_stub(calls, behavior):
    class M:
        def __init__(self, name):
            self.name = name

        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, prompt):
            calls.append(self.name)
            return behavior(self.name)

    return M


def test_falls_back_to_second_model(monkeypatch):
    from task_organizer import llm as llm_module

    monkeypatch.setenv("GROK_MODEL", "primary")
    monkeypatch.setenv("GROK_FALLBACK_MODELS", "backup")
    calls = []

    def behavior(name):
        if name == "primary":
            raise RuntimeError("503 UNAVAILABLE")
        return "ok"

    M = _model_stub(calls, behavior)
    monkeypatch.setattr(llm_module, "get_llm", lambda model=None: M(model))
    assert llm_module.invoke_structured(object, "p") == "ok"
    assert calls == ["primary"] * llm_module.RETRIES_PER_MODEL + ["backup"]


def test_missing_model_skips_to_next_without_retrying(monkeypatch):
    from task_organizer import llm as llm_module

    monkeypatch.setenv("GROK_MODEL", "old")
    monkeypatch.setenv("GROK_FALLBACK_MODELS", "new")
    calls = []

    def behavior(name):
        if name == "old":
            raise RuntimeError("404 NOT_FOUND model no longer available")
        return "ok"

    M = _model_stub(calls, behavior)
    monkeypatch.setattr(llm_module, "get_llm", lambda model=None: M(model))
    assert llm_module.invoke_structured(object, "p") == "ok"
    assert calls == ["old", "new"]


def test_fatal_error_is_not_retried(monkeypatch):
    import pytest

    from task_organizer import llm as llm_module

    monkeypatch.setenv("GROK_MODEL", "primary")
    monkeypatch.delenv("GROK_FALLBACK_MODELS", raising=False)
    calls = []

    def behavior(name):
        raise ValueError("400 INVALID_ARGUMENT bad request")

    M = _model_stub(calls, behavior)
    monkeypatch.setattr(llm_module, "get_llm", lambda model=None: M(model))
    with pytest.raises(ValueError):
        llm_module.invoke_structured(object, "p")
    assert calls == ["primary"]
