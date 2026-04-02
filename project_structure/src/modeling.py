from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from src.config import OHE_COLS, ORDINAL_COLS
from src.preprocessing import DepressionFeatureEngineer

def build_pipeline():
    """Builds the full scikit-learn pipeline."""
    
    # 1. Encoders
    ordinal_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    ohe_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 2. Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('ord_pipeline', ordinal_transformer, ORDINAL_COLS),
            ('ohe_pipeline', ohe_transformer, OHE_COLS),
        ],
        remainder='passthrough',
        n_jobs=1
    )
    
    # 3. Model with your Optuna best parameters
    best_params = {
        'n_estimators': 600, 
        'max_depth': 3, 
        'learning_rate': 0.0309, 
        'subsample': 0.603, 
        'colsample_bytree': 0.760
    }
    xgb_model = XGBClassifier(**best_params, random_state=42)
    
    # 4. Final Pipeline
    pipeline = Pipeline(steps=[
        ('feature_engineering', DepressionFeatureEngineer()),
        ('encoding', preprocessor),
        ('classifier', xgb_model)
    ])
    
    return pipeline