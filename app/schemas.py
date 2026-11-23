"""Request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .stats import DIM


class FactoryPayload(BaseModel):
    """One factory's integer summary.

    sums  = [sum speed, sum pressure, sum power]   (integers)
    outer = 3x3 integer matrix of sum(x_i * x_j) for x=(speed, pressure, power)
    correction = integer zero-point correction added to the pressure.
    """

    factory_id: str = Field(min_length=1, max_length=64)
    n: int = Field(ge=1)
    sums: list[int] = Field(min_length=DIM, max_length=DIM)
    outer: list[list[int]] = Field(min_length=DIM, max_length=DIM)
    correction: int = 0

    @field_validator("outer")
    @classmethod
    def _square(cls, v):
        if any(len(row) != DIM for row in v):
            raise ValueError("outer must be a 3x3 matrix")
        return v


class CreateVersionRequest(BaseModel):
    factories: list[FactoryPayload] = Field(min_length=3, max_length=8)

    @field_validator("factories")
    @classmethod
    def _unique_ids(cls, v):
        ids = [f.factory_id for f in v]
        if len(set(ids)) != len(ids):
            raise ValueError("factory_id values must be unique")
        return v


class CorrectionUpdate(BaseModel):
    factory_id: str = Field(min_length=1, max_length=64)
    correction: int


class PredictRequest(BaseModel):
    factory_id: str = Field(min_length=1, max_length=64)
    speed: float
    pressure: float
    # Optional pilot batch: predictions then use the batch's pinned model
    # version, zero-point correction and power offset.
    batch_id: str | None = Field(default=None, min_length=1, max_length=64)


class OpenBatchRequest(BaseModel):
    """Open (or re-open) a pilot batch for one factory.

    Re-opening the same (factory_id, batch_id) returns the original pins
    unchanged, so a retried request is idempotent.
    """

    factory_id: str = Field(min_length=1, max_length=64)
    batch_id: str = Field(min_length=1, max_length=64)


class ReportPayload(BaseModel):
    """One trial-run operating summary reported for a pilot batch.

    Same integer sufficient statistics as a factory payload; the batch
    supplies version / correction / offset, so they are not accepted here.
    """

    n: int = Field(ge=1)
    sums: list[int] = Field(min_length=DIM, max_length=DIM)
    outer: list[list[int]] = Field(min_length=DIM, max_length=DIM)

    @field_validator("outer")
    @classmethod
    def _square(cls, v):
        if any(len(row) != DIM for row in v):
            raise ValueError("outer must be a 3x3 matrix")
        return v
