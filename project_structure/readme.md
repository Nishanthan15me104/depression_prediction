

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