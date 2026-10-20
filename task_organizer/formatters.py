"""Render the final task list as text, markdown, or JSON."""

import json
from typing import List

PRIORITY_ORDER = ("High", "Medium", "Low")
ICONS = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
EMPTY_MESSAGE = "No tasks found in your message."

DEFAULT_PRIORITY = "Medium"
DEFAULT_REASON = "Could not be prioritized automatically; defaulted to Medium."


def sort_tasks(tasks: List[dict]) -> List[dict]:
    """Sort by priority (High first), then by original order."""
    rank = {p: i for i, p in enumerate(PRIORITY_ORDER)}
    return sorted(tasks, key=lambda t: (rank.get(t["priority"], 99), t["id"]))


def _group(tasks: List[dict]):
    for level in PRIORITY_ORDER:
        group = [t for t in tasks if t["priority"] == level]
        if group:
            yield level, group


def render_text(tasks: List[dict]) -> str:
    if not tasks:
        return EMPTY_MESSAGE
    lines = ["📋 YOUR ORGANIZED TO-DO LIST", "=" * 32]
    for level, group in _group(tasks):
        lines.append(f"\n{ICONS[level]} {level.upper()} PRIORITY")
        for t in group:
            due = f" (due: {t['deadline']})" if t.get("deadline") else ""
            lines.append(f"  [ ] {t['title']}{due}")
            lines.append(f"      ↳ {t['reason']}")
    return "\n".join(lines)


def render_markdown(tasks: List[dict]) -> str:
    if not tasks:
        return EMPTY_MESSAGE
    lines = ["# To-Do List"]
    for level, group in _group(tasks):
        lines.append(f"\n## {ICONS[level]} {level} priority\n")
        for t in group:
            due = f" *(due: {t['deadline']})*" if t.get("deadline") else ""
            lines.append(f"- [ ] **{t['title']}**{due}: {t['reason']}")
    return "\n".join(lines) + "\n"


def render_json(tasks: List[dict]) -> str:
    return json.dumps(tasks, indent=2, ensure_ascii=False)


RENDERERS = {"text": render_text, "markdown": render_markdown, "json": render_json}
