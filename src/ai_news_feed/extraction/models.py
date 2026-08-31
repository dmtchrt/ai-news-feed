"""Typed extraction boundary consumed by the normalizer."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ExtractionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExtractedItem(ExtractionModel):
    text: str = Field(min_length=1)
    title: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ExtractionFailure(ExtractionModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
