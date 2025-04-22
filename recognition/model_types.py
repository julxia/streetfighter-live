from pydantic import BaseModel


class ModelOutput(BaseModel):
    output: str | None
    confidence: float
