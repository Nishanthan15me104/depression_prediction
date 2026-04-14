

```
depression_prediction/
├── data/
│   ├── raw/                 # Put your train.csv and test.csv here
│   └── processed/           # Save outputs/submissions here
├── src/
│   ├── __init__.py
│   ├── config.py            # Global variables and paths
│   ├── preprocessing.py     # Custom Scikit-Learn transformers for feature engineering
│   ├── modeling.py          # Model pipeline and evaluation
├── requirements.txt         # Dependencies
└── main.py                  # Entry point to run the pipeline
```

depression_prediction/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── config.py       # Phase 6: Config + Env
│   │   ├── security.py     # Phase 5: Security (AuthN/AuthZ)
│   │   ├── schemas.py      # Phase 2: Request Validation (Pydantic)
│   │   ├── services.py     # Phase 3 & 4: Service Layer & Async
│   │   ├── routes.py       # Phase 1: Routing
│   │   └── main.py         # Phase 1 & 9: Setup, Middleware, Execution