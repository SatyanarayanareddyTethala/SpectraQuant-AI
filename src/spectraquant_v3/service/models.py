"""Typed API contracts for the SpectraQuant V3 control plane."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


ExecutionMode = Literal["research", "paper", "live"]
AssetClass = Literal["crypto", "equity"]
RunState = Literal["queued", "running", "cancelling", "cancelled", "success", "failed", "timed_out"]


class RunSubmissionRequest(BaseModel):
    asset_class: AssetClass
    execution_mode: ExecutionMode = "research"
    run_mode: Literal["normal", "test", "refresh"] = "normal"
    dry_run: bool = False
    idempotency_key: str = Field(min_length=8, max_length=128)
    config_dir: str | None = None
    run_id: str | None = None
    approval_token: str | None = None


class StageEvent(BaseModel):
    stage: str
    status: Literal["started", "ok", "failed"]
    at: datetime
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    run_id: str
    idempotency_key: str
    asset_class: AssetClass
    execution_mode: ExecutionMode
    run_mode: str
    dry_run: bool
    state: RunState
    created_at: datetime
    updated_at: datetime
    error_code: str = ""
    error_message: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    cancellation_requested_at: datetime | None = None
    terminal_at: datetime | None = None
    terminal_reason: str = ""


class ApiEnvelope(BaseModel):
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, Any] = Field(default_factory=dict)
