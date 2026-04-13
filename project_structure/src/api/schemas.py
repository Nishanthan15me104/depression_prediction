"""
Pydantic schemas for the FastAPI application.
Enforces strict input validation and type checking before data reaches the model.
"""
from pydantic import BaseModel, Field
from typing import Optional

class DepressionPredictionInput(BaseModel):
    """
    Pydantic model defining the expected JSON payload for a single prediction.
    Includes all 18 fields required by the MLflow model signature.
    """
    Name: Optional[str] = Field(default="Unknown")
    Gender: str = Field(description="Gender of the individual")
    Age: float = Field(description="Age of the individual")
    City: Optional[str] = Field(default="Unknown")
    Working_Professional_or_Student: str = Field(alias="Working Professional or Student")
    Profession: Optional[str] = Field(default=None)
    Academic_Pressure: Optional[float] = Field(alias="Academic Pressure", default=None)
    Work_Pressure: Optional[float] = Field(alias="Work Pressure", default=None)
    CGPA: Optional[float] = Field(default=None)
    Study_Satisfaction: Optional[float] = Field(alias="Study Satisfaction", default=None)
    Job_Satisfaction: Optional[float] = Field(alias="Job Satisfaction", default=None)
    Sleep_Duration: str = Field(alias="Sleep Duration")
    Dietary_Habits: str = Field(alias="Dietary Habits")
    Degree: Optional[str] = Field(default=None)
    Suicidal_Thoughts: str = Field(alias="Have you ever had suicidal thoughts ?")
    Work_Study_Hours: float = Field(alias="Work/Study Hours")
    Financial_Stress: float = Field(alias="Financial Stress")
    Family_History: str = Field(alias="Family History of Mental Illness")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Name": "Sanya",
                "Gender": "Female",
                "Age": 58.0,
                "City": "Kolkata",
                "Working Professional or Student": "Working Professional",
                "Profession": "Educational Consultant",
                "Academic Pressure": None,
                "Work Pressure": 2.0,
                "CGPA": None,
                "Study Satisfaction": None,
                "Job Satisfaction": 4.0,
                "Sleep Duration": "Less than 5 hours",
                "Dietary Habits": "Moderate",
                "Degree": "B.Ed",
                "Have you ever had suicidal thoughts ?": "No",
                "Work/Study Hours": 6.0,
                "Financial Stress": 4.0,
                "Family History of Mental Illness": "No"
            }
        }