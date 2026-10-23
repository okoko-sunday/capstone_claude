"""Request/response contracts for the HTTP API (also drive /docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["cat", "dog", "neither"]


class PredictionOut(BaseModel):
    filename: str
    label: Literal["cat", "dog", "neither", "error"]
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    decided_by: Literal["model", "threshold"]
    error: str | None = None


class PredictResponse(BaseModel):
    predictions: list[PredictionOut]
    counts: dict[str, int]
    elapsed_ms: float


class EvalItem(BaseModel):
    true: Label
    predicted: Literal["cat", "dog", "neither", "error"]


class EvaluateRequest(BaseModel):
    items: list[EvalItem] = Field(min_length=1, max_length=10_000)


class ClassReportOut(BaseModel):
    label: str
    support: int
    predicted: int
    precision: float | None
    recall: float | None
    f1: float | None


class EvaluateResponse(BaseModel):
    labels: list[str]
    matrix: list[list[int]]
    total: int
    correct: int
    accuracy: float | None
    per_class: list[ClassReportOut]
    macro_f1: float | None
    errors_excluded: int


class HealthResponse(BaseModel):
    status: Literal["ok", "no_model"]
    model_loaded: bool
    detail: str | None = None
    version: str
