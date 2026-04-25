import requests
import time

url = "http://127.0.0.1:8000/api/v1/predict"
headers = {"X-API-Key": "my-super-secret-production-key"}

# Sample data based on your schema
payload = {
    "Gender": "Male",
    "Age": 25,
    "Working Professional or Student": "Student",
    "Sleep Duration": "7-8 hours",
    "Dietary Habits": "Healthy",
    "Have you ever had suicidal thoughts ?": "No",
    "Work/Study Hours": 8.0,
    "Financial Stress": 2.0,
    "Family History of Mental Illness": "No"
}

print(f"🚀 Sending 100 requests to {url}...")

for i in range(100):
    start = time.time()
    response = requests.post(url, json=payload, headers=headers)
    duration = time.time() - start
    if i % 10 == 0:
        print(f"Request {i}: Status {response.status_code} | Latency: {duration:.4f}s")

print("✅ Baseline traffic complete.")