
folder structure
```
depression_prediction/
├── data/
│   ├── raw/                 # Put your train.csv and test.csv here
│   └── processed/           # Save outputs/submissions here
├── src/
│   ├── api/
│   │   ├── static_models/
│   │   │   └── final_model.pkl  <-- Created by the script
│   │   ├── __init__.py
│   │   ├── config.py       # Phase 6: Config + Env
│   │   ├── security.py     # Phase 5: Security (AuthN/AuthZ)
│   │   ├── schemas.py      # Phase 2: Request Validation (Pydantic)
│   │   ├── services.py     # Phase 3 & 4: Service Layer & Async
│   │   ├── routes.py       # Phase 1: Routing
│   │   └── main.py         # Phase 1 & 9: Setup, Middleware, Execution
│   ├── __init__.py
│   ├── config.py            # Global variables and paths
│   ├── preprocessing.py     # Custom Scikit-Learn transformers for feature engineering
│   ├── modeling.py          # Model pipeline and evaluation
├── __init__.py
├── .dockerignore
├── .env
├── .gitignore
├── comments.md
├── docker-compose.yml
├── Dockerfile
├── export_model.py
├── main.py
├── mlflow.db
├── requirements.txt         # Dependencies
└── runapi.py
```



challenges an problwm solving:
 - use export fiel to store machine learin moel as pkl fiel and upload to fiel 
 - insted of usin ACR(Azure) use gitub ghct to deploy in acr 

 - To take the API from a local development state to a production-grade Docker container, we decoupled the service from MLflow by loading the model from a static .pkl file and implemented a manual data-mapping bridge in the routes. This ensures the raw model output is correctly wrapped into your standardized JSON schema, preventing validation errors while remaining lightweight and self-contained.
3. Tight Coupling vs. Decoupling The Challenge: Your local development environment was "leaking" into your production path. A 500MB library (MLflow) was a "Hard Dependency," meaning your production API couldn't live without its development "parent." The Fix: You achieved Architectural Decoupling. You created a clear separation where: DEV Path: Uses all the heavy bells and whistles (MLflow, tracking, etc.). PROD Path: Is lean, fast, and only uses the bare minimum (Joblib, Pandas, Scikit-Learn).