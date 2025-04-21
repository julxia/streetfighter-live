from pydantic import BaseModel


class ModelOutput(BaseModel):
    output: str
    confidence: float
