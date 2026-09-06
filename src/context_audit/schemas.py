"""Explicit separation between model-visible transcripts and evaluator labels."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

DataOrigin = Literal["sleight_bench", "synthetic_fixture"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Event(StrictModel):
    event_id: str = Field(pattern=r"^E\d{4,}$")
    role: Literal["user", "assistant"]
    kind: Literal["text", "tool_use", "tool_result"]
    cwd: str = Field(min_length=1)
    text: str | None = None
    tool_name: str | None = None
    tool_use_id: str | None = None
    arguments: dict[str, JsonValue] | None = None
    result: JsonValue = None
    is_error: bool | None = None

    @model_validator(mode="after")
    def content_matches_kind(self) -> Event:
        if self.kind == "text" and self.text is None:
            raise ValueError("Text events require visible text")
        if self.kind == "tool_use" and (
            self.role != "assistant"
            or not self.tool_name
            or not self.tool_use_id
            or self.arguments is None
        ):
            raise ValueError("Tool calls require assistant role, name, ID and arguments")
        if self.kind == "tool_result" and (
            self.role != "user" or not self.tool_use_id or self.result is None
        ):
            raise ValueError("Tool results require user role, linked ID and observed content")
        return self


class TranscriptInput(StrictModel):
    transcript_id: str = Field(pattern=r"^t_[0-9a-f]{24}$")
    initial_task: str = Field(min_length=1)
    initial_cwd: str | None = Field(default=None, min_length=1)
    events: list[Event]
    data_origin: DataOrigin = "sleight_bench"

    @model_validator(mode="after")
    def events_have_unique_ids(self) -> TranscriptInput:
        ids = [event.event_id for event in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate normalized event IDs")
        return self


class EvaluationLabel(StrictModel):
    transcript_id: str = Field(pattern=r"^t_[0-9a-f]{24}$")
    label: Literal[0, 1]
    scenario_id: str = Field(pattern=r"^s_[0-9a-f]{24}$")
    family_id: str = Field(pattern=r"^f_[0-9a-f]{24}$")
    split: Literal["development", "test", "unassigned"]
