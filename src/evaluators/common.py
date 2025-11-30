# src/evaluators/common.py

from dataclasses import dataclass
from typing import Protocol

# ---- Response Protocol ----
class ModelResponse(Protocol):
    @property
    def text(self) -> str:
        ...

# ---- Dummy Response ----
@dataclass(frozen=True)
class DummyResponse:
    _text: str

    @property
    def text(self) -> str:
        return self._text


# ---- Model Protocol ----
class ModelClient(Protocol):
    def complete(self, prompt: str) -> ModelResponse:
        """
        Expected interface for any model-like class.
        """
        ...


# ---- Dummy Model ----
class DummyModelClient(ModelClient):
    def complete(self, prompt: str) -> DummyResponse:
        # Simple deterministic fake score for bias eval
        return DummyResponse("7.0")

