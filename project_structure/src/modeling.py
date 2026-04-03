from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from src.config import OHE_COLS, ORDINAL_COLS
from src.preprocessing import DepressionFeatureEngineer

def build_pipeline(xgb_params=None):
    """Builds the full scikit-learn pipeline with dynamic parameters."""
    
    # 1. Encoders
    ordinal_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    ohe_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True)) # Set to True for memory efficiency
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
    
    # 3. Model
    if xgb_params is None:
        # Default fallback params
        xgb_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}
        
    xgb_model = XGBClassifier(**xgb_params, random_state=42)
    
    # 4. Final Pipeline
    pipeline = Pipeline(steps=[
        ('feature_engineering', DepressionFeatureEngineer()),
        ('encoding', preprocessor),
        ('classifier', xgb_model)
    ])
    
    return pipeline