import requests
import time

url = "http://127.0.0.1:8000/api/v1/predict"
headers = {"X-API-Key": "dev-secret-key-1234"}

# Full payload matching your schema
payload = {
    "Name": "Test User",
    "Gender": "Male",
    "Age": 25.0,
    "City": "Chennai",
    # Note: Use the Alias names (with spaces) as they appear in the schema
    "Working Professional or Student": "Student",
    "Profession": "None",
    "Academic Pressure": 3.0,  # Must be 1-5
    "Work Pressure": 1.0,      # FIXED: Changed from 0.0 to 1.0 (to meet ge=1)
    "CGPA": 8.0,               # Must be 0-10
    "Study Satisfaction": 4.0, # Must be 1-5
    "Job Satisfaction": 1.0,   # FIXED: Changed from 0.0 to 1.0 (to meet ge=1)
    "Sleep Duration": "7-8 hours",
    "Dietary Habits": "Healthy",
    "Degree": "B.E",
    "Have you ever had suicidal thoughts ?": "No",
    "Work/Study Hours": 8.0,   # Must be 0-24
    "Financial Stress": 2.0,   # Must be 1-5
    "Family History of Mental Illness": "No"
}

print(f"🚀 Sending 100 requests to {url}...")

for i in range(100):
    response = requests.post(url, json=payload, headers=headers)
    if i % 20 == 0:
        print(f"Request {i}: Status {response.status_code}")

print("✅ Buffer is now full with complete data.")