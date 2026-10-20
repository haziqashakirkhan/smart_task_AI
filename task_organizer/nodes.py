"""LangGraph nodes: extraction, prioritization, formatting, plus routers."""

from datetime import date
from typing import Optional

from .formatters import (
    DEFAULT_PRIORITY,
    DEFAULT_REASON,
    render_text,
    sort_tasks,
)
from .llm import invoke_structured
from .schemas import ExtractedTasks, PrioritizedTasks
from .state import TaskState

MAX_PRIORITIZE_ATTEMPTS = 3
_NULL_WORDS = {"", "null", "none", "n/a", "na", "unspecified"}


def _clean_deadline(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return None if value.lower() in _NULL_WORDS else value


# --------------------------------------------------------------------------
# Node 1: Task extraction
# --------------------------------------------------------------------------
def extract_tasks(state: TaskState) -> TaskState:
    prompt = f"""Today's date is {date.today().isoformat()}.

Extract every individual task from the message below.
- Split combined sentences into separate tasks.
- Keep titles short and actionable (start with a verb when possible).
- Capture any deadline the user mentioned, exactly as phrased; use null if none.
- Do not invent tasks that are not in the message.

Message:
{state["user_input"]}"""
    result: ExtractedTasks = invoke_structured(ExtractedTasks, prompt)

    tasks = []
    for t in result.tasks:
        title = t.title.strip()
        if title:
            tasks.append(
                {"id": len(tasks) + 1, "title": title, "deadline": _clean_deadline(t.deadline)}
            )
    return {"tasks": tasks, "prioritized_tasks": [], "attempts": 0}


# --------------------------------------------------------------------------
# Node 2: Task prioritization (re-runs only for tasks still missing)
# --------------------------------------------------------------------------
def prioritize_tasks(state: TaskState) -> TaskState:
    already = list(state.get("prioritized_tasks", []))
    done_ids = {t["id"] for t in already}
    pending = [t for t in state["tasks"] if t["id"] not in done_ids]
    attempts = state.get("attempts", 0) + 1
    if not pending:
        return {"attempts": attempts}

    task_lines = "\n".join(
        f"[{t['id']}] {t['title']} (deadline: {t['deadline'] or 'none'})" for t in pending
    )
    prompt = f"""Today's date is {date.today().isoformat()}.

Assign each task a priority: High, Medium, or Low.
Base it mainly on deadline urgency, then on importance and the consequences of
missing it. Return exactly one item per task, using the id shown in brackets.

Tasks:
{task_lines}"""
    result: PrioritizedTasks = invoke_structured(PrioritizedTasks, prompt)

    pending_by_id = {t["id"]: t for t in pending}
    for item in result.items:
        task = pending_by_id.pop(item.id, None)  # ignores unknown/duplicate ids
        if task is not None:
            already.append(
                {**task, "priority": item.priority, "reason": item.reason.strip()}
            )
    return {"prioritized_tasks": already, "attempts": attempts}


# --------------------------------------------------------------------------
# Node 3: Format results (plain Python)
# --------------------------------------------------------------------------
def format_results(state: TaskState) -> TaskState:
    done = {t["id"]: t for t in state.get("prioritized_tasks", [])}
    final = []
    for t in state.get("tasks", []):
        final.append(
            done.get(t["id"])
            or {**t, "priority": DEFAULT_PRIORITY, "reason": DEFAULT_REASON}
        )
    final = sort_tasks(final)
    return {"final_tasks": final, "formatted_output": render_text(final)}


# --------------------------------------------------------------------------
# Conditional routers
# --------------------------------------------------------------------------
def route_after_extract(state: TaskState) -> str:
    return "prioritize_tasks" if state.get("tasks") else "format_results"


def route_after_prioritize(state: TaskState) -> str:
    all_done = len(state.get("prioritized_tasks", [])) >= len(state.get("tasks", []))
    if all_done or state.get("attempts", 0) >= MAX_PRIORITIZE_ATTEMPTS:
        return "format_results"
    return "prioritize_tasks"
