# 🧠 Depression Prediction API

A modular, production-ready machine learning API designed to predict depression indicators based on demographic, academic, and lifestyle features.

This project features:
- ✅ Scikit-learn + XGBoost pipeline
- ✅ FastAPI backend
- ✅ Production-ready architecture
- ✅ Dockerized deployment
- ✅ Prometheus + Grafana observability
- ✅ CI/CD with GitHub Actions
- ✅ Static optimized model serving

---

# 📁 Directory Structure

```text
depression_prediction/
├── data/
│   ├── raw/                       # Put train.csv and test.csv here
│   └── processed/                 # Save outputs/submissions here
│
├── src/
│   ├── api/
│   │   ├── static_models/
│   │   │   └── final_model.pkl    # Exported production model
│   │   │
│   │   ├── __init__.py
│   │   ├── config.py              # Environment + configuration
│   │   ├── security.py            # API authentication
│   │   ├── schemas.py             # Pydantic request validation
│   │   ├── services.py            # Business/service layer
│   │   ├── routes.py              # API routes
│   │   └── main.py                # FastAPI setup + middleware
│   │
│   ├── __init__.py
│   ├── config.py                  # Global variables and paths
│   ├── preprocessing.py           # Feature engineering transformers
│   └── modeling.py                # Model training/evaluation pipeline
│
├── __init__.py
├── .dockerignore
├── .env
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── export_model.py
├── main.py
├── mlflow.db
├── requirements-prod.txt
└── run_api.py
```

---

# 🧠 Architecture & Deep Dive

This project follows a strict **separation of concerns** to ensure:
- scalability
- maintainability
- modularity
- production readiness

---

## 1️⃣ Presentation & API Layer (`src/api/`)

Built using :contentReference[oaicite:0]{index=0} for high-performance REST APIs.

### Responsibilities
- API routing
- Request validation
- Security/authentication
- Response standardization
- Async request handling

### Core Components

| File | Responsibility |
|---|---|
| `routes.py` | API endpoints |
| `schemas.py` | Pydantic request validation |
| `security.py` | API key validation |
| `services.py` | Prediction business logic |
| `main.py` | Middleware + startup configuration |

---

## 2️⃣ Machine Learning Pipeline (`src/`)

Uses a modular :contentReference[oaicite:1]{index=1} pipeline with :contentReference[oaicite:2]{index=2} for prediction.

### Pipeline Features
- Custom transformers
- Feature engineering
- Missing value handling
- Target encoding
- Leakage prevention
- Modular preprocessing

### Files

| File | Responsibility |
|---|---|
| `preprocessing.py` | Feature engineering |
| `modeling.py` | Training + evaluation |

---

## 3️⃣ Observability & Monitoring

The system is fully containerized with monitoring support.

### Stack
- :contentReference[oaicite:3]{index=3} → Metrics scraping
- :contentReference[oaicite:4]{index=4} → Visualization dashboards
- Docker Compose orchestration

### Metrics Monitored
- API latency
- Throughput
- Request count
- Status codes
- Memory usage
- Resource behavior

---

## 4️⃣ CI/CD & Deployment

Automated CI/CD pipeline using:
- :contentReference[oaicite:5]{index=5} Actions
- Docker Buildx
- GHCR (GitHub Container Registry)

### CI/CD Workflow
1. Push code to GitHub
2. GitHub Actions triggers workflow
3. Docker image builds automatically
4. Image pushed to GHCR
5. Ready for deployment

---

# 🛠️ Challenges & Problem Solving

---

## 1️⃣ Tight Coupling vs Decoupling

### ❌ Challenge

The development environment was tightly coupled with production:
- MLflow became a hard dependency
- Added ~500MB overhead
- Production API depended on development tracking environment

### ✅ Solution

Architectural decoupling.

### DEV Path
Uses MLflow for:
- experiment tracking
- logging
- evaluation
- training comparison

### PROD Path
Optimized lightweight serving:
- Static `.pkl` loading via `joblib`
- Exported using `export_model.py`
- Lightweight inference path
- No MLflow dependency in production

### Additional Optimization
Implemented manual data-mapping bridge in API routes:
- Converts raw model output
- Wraps response into standardized JSON schema
- Maintains response consistency

---

## 2️⃣ Container Registry Migration

### ❌ Challenge

Avoid unnecessary external cloud overhead for container hosting.

### ✅ Solution

Migrated from:
- Azure Container Registry (ACR)

To:
- GitHub Container Registry (GHCR)

### Benefits
- Fully GitHub-native workflow
- Simplified CI/CD
- Reduced external dependencies
- Easier maintenance

---

# ⚙️ Setup Instructions

---

## 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/depression_prediction.git

cd depression_prediction
```

---

## 2️⃣ Create Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### Linux / Mac

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements-prod.txt
```

---

## 4️⃣ Configure Environment Variables

Create `.env` file:

```env
APP_ENV=PROD

API_KEY=dev-secret-key-1234
```

---

# 🚀 Running the Application

---

## Option A — Docker Deployment (Recommended)

Runs:
- API
- Prometheus
- Grafana

```bash
docker-compose up --build -d
```

---

## Option B — Local Execution

```bash
python run_api.py
```

---

# 📡 API Endpoints

Application runs at:

```text
http://127.0.0.1:8000
```

---

# ▶ 1️⃣ Health Check Endpoint

Verify API is live.

## Endpoint

```http
GET /
```

## Response

```json
{
    "status": "ok",
    "message": "API is running"
}
```

---

# ▶ 2️⃣ Prediction Endpoint

Predict depression indicators using user demographic + lifestyle data.

## Endpoint

```http
POST /api/v1/predict
```

---

## Headers

```http
Content-Type: application/json

X-API-Key: dev-secret-key-1234
```

---

## Example Payload

```json
{
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
```

---

# 📊 Performance Metrics — v1.1.0

**Date:** 2026-04-25  
**Environment:** Local (Windows, Python 3.11.6)  
**Mode:** `APP_ENV=DEV`  
**Model Loading:** Static `.pkl` via `joblib`  
**Test Type:** 100 Consecutive POST Requests  

---

## 📈 Latency Metrics

*(Source: `http_request_duration_highr_seconds`)*

| Metric | Value |
|---|---|
| Mean Latency | 35.43 ms |
| P50 Latency | < 50 ms |
| P97 Latency | 50 ms |
| P99 Latency | 75 ms |
| Max Latency | 100 ms |

### Breakdown
- 97% of requests completed under 50ms
- 99% completed under 75ms
- All requests completed within 100ms

---

## 📦 Throughput Metrics

| Metric | Value |
|---|---|
| Success Rate | 100% |
| Avg Request Size | 274 bytes |
| Avg Response Size | 126 bytes |

---

## 🛠️ Python Garbage Collection Metrics

| Metric | Value |
|---|---|
| Generation 0 Collections | 471 |
| Objects Collected | 9,226 |

### Note
High GC activity is expected during burst traffic because Python rapidly creates and destroys:
- JSON payloads
- dictionaries
- request objects
- response objects

Memory remained stable under rapid inference load.

---

# ✅ Key Achievements

- ⚡ Average inference latency around 35ms
- 🚀 P99 latency under 75ms
- 💯 100% successful requests
- 📉 Efficient memory handling
- 🧠 Production-ready modular architecture
- 📦 Lightweight static model deployment
- 📊 Integrated observability stack
- 🔄 CI/CD automated deployment pipeline

---

# 🔮 Future Improvements

- Async benchmarking under concurrent load
- Kubernetes deployment
- Redis request caching
- Distributed tracing
- Auto-scaling infrastructure
- Canary deployment strategy
- Advanced monitoring alerts

---

# 👨‍💻 Tech Stack

| Category | Technologies |
|---|---|
| Backend | Python, FastAPI |
| ML | Scikit-learn, XGBoost |
| Monitoring | Prometheus, Grafana |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Registry | GHCR |
| Validation | Pydantic |

---