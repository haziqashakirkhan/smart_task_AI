"""Shared state passed between LangGraph nodes."""

from typing import List, Optional, TypedDict


class TaskState(TypedDict, total=False):
    user_input: str
    # [{"id": 1, "title": "...", "deadline": "tomorrow" | None}]
    tasks: List[dict]
    # tasks above plus "priority" and "reason"
    prioritized_tasks: List[dict]
    attempts: int
    # Final, sorted list (priority order) with defaults filled in
    final_tasks: List[dict]
    formatted_output: str
