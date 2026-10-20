"""Assemble the LangGraph workflow.

START -> extract_tasks -> prioritize_tasks (retries if incomplete) -> format_results -> END
"""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    extract_tasks,
    format_results,
    prioritize_tasks,
    route_after_extract,
    route_after_prioritize,
)
from .state import TaskState


def build_graph():
    graph = StateGraph(TaskState)
    graph.add_node("extract_tasks", extract_tasks)
    graph.add_node("prioritize_tasks", prioritize_tasks)
    graph.add_node("format_results", format_results)

    graph.add_edge(START, "extract_tasks")
    graph.add_conditional_edges(
        "extract_tasks",
        route_after_extract,
        {"prioritize_tasks": "prioritize_tasks", "format_results": "format_results"},
    )
    graph.add_conditional_edges(
        "prioritize_tasks",
        route_after_prioritize,
        {"prioritize_tasks": "prioritize_tasks", "format_results": "format_results"},
    )
    graph.add_edge("format_results", END)
    return graph.compile()


def organize_tasks(user_input: str) -> dict:
    """Run the full workflow and return the final state."""
    if not user_input or not user_input.strip():
        raise ValueError("Please provide some text describing your tasks.")
    return build_graph().invoke({"user_input": user_input.strip()})
