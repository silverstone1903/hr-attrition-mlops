
from typing import Any, Optional, Union
from pydantic import BaseModel


class TrainApiData(BaseModel):
    model_name: str
    hyperparams: dict


class PredictApiData(BaseModel):
    model_name: str
    data: Any
    