"""
Pydantic schemas for the FastAPI application.
Enforces strict input validation and type checking.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class DepressionPredictionInput(BaseModel):
    Name: Optional[str] = Field(default="Unknown")
    Gender: str = Field(description="Gender of the individual")
    Age: float = Field(description="Age of the individual", ge=0, le=120)
    City: Optional[str] = Field(default="Unknown")
    Working_Professional_or_Student: str = Field(alias="Working Professional or Student")
    Profession: Optional[str] = Field(default=None)
    Academic_Pressure: Optional[float] = Field(alias="Academic Pressure", default=None, ge=1, le=5)
    Work_Pressure: Optional[float] = Field(alias="Work Pressure", default=None, ge=1, le=5)
    CGPA: Optional[float] = Field(default=None, ge=0, le=10)
    Study_Satisfaction: Optional[float] = Field(alias="Study Satisfaction", default=None, ge=1, le=5)
    Job_Satisfaction: Optional[float] = Field(alias="Job Satisfaction", default=None, ge=1, le=5)
    Sleep_Duration: str = Field(alias="Sleep Duration")
    Dietary_Habits: str = Field(alias="Dietary Habits")
    Degree: Optional[str] = Field(default=None)
    Suicidal_Thoughts: str = Field(alias="Have you ever had suicidal thoughts ?")
    Work_Study_Hours: float = Field(alias="Work/Study Hours", ge=0, le=24)
    Financial_Stress: float = Field(alias="Financial Stress", ge=1, le=5)
    Family_History: str = Field(alias="Family History of Mental Illness")

    model_config = ConfigDict(populate_by_name=True)

# THIS WAS MISSING: The output schema required by routes.py
class PredictionOutput(BaseModel):
    """Standardized output response for the API."""
    status: str
    prediction: int
    probability_0: float
    probability_1: float
    message: str