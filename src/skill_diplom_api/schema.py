from pydantic import BaseModel, Field


class ModelTrain(BaseModel):
    Data: list[dict]


class ModelOutp(BaseModel):
    Training_time: str
    Status: str
    Message: str
    precision_at_3: str
    recall_at_3: str


class ForecastOutp(BaseModel):
    Forecast_time: str
    Status: str
    Recommendations: dict


