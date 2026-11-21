from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Person(BaseModel):
    """One applicant. Field names/values follow the UCI Adult dataset."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 39, "education_num": 13, "capital_gain": 2174, "capital_loss": 0,
                "hours_per_week": 40, "workclass": "State-gov", "marital_status": "Never-married",
                "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White",
                "sex": "Male", "native_country": "United-States",
            }
        }
    )

    age: int = Field(ge=16, le=110)
    education_num: int = Field(ge=1, le=16, description="Years of education (1-16)")
    capital_gain: float = Field(ge=0, le=1_000_000)
    capital_loss: float = Field(ge=0, le=1_000_000)
    hours_per_week: int = Field(ge=1, le=100)
    workclass: str = Field(min_length=1, max_length=60)
    marital_status: str = Field(min_length=1, max_length=60)
    occupation: str = Field(min_length=1, max_length=60)
    relationship: str = Field(min_length=1, max_length=60)
    race: str = Field(min_length=1, max_length=60)
    sex: Literal["Male", "Female"]
    native_country: str = Field(min_length=1, max_length=60)


class PredictRequest(BaseModel):
    instances: list[Person] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    probability: float
    prediction: int
    label: str


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    predictions: list[Prediction]
    latency_ms: float


class Contribution(BaseModel):
    feature: str
    value: Any
    contribution: float


class ExplainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    probability: float
    prediction: int
    label: str
    base_probability: float
    contributions: list[Contribution]
    method: str


class CounterfactualResponse(BaseModel):
    """What happens to the prediction if ONLY the protected attribute is flipped?"""

    original: Prediction
    flipped: Prediction
    original_sex: str
    flipped_sex: str
    probability_change: float
    decision_changed: bool
    interpretation: str
