"""Pydantic schemas the model must fill in (structured output)."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Priority = Literal["High", "Medium", "Low"]


class ExtractedTask(BaseModel):
    title: str = Field(description="Short, actionable task title, e.g. 'Buy groceries'")
    deadline: Optional[str] = Field(
        default=None,
        description="Deadline as the user stated it (e.g. 'tomorrow', 'next week'), or null if none",
    )


class ExtractedTasks(BaseModel):
    tasks: List[ExtractedTask]


class PriorityItem(BaseModel):
    id: int = Field(description="The id of the task exactly as given in the prompt")
    priority: Priority
    reason: str = Field(description="One short sentence justifying the priority")


class PrioritizedTasks(BaseModel):
    items: List[PriorityItem]
