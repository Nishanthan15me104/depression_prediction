"""
Main execution script for data loading, Optuna tuning, and MLflow tracking.
"""
import os
import time
import hashlib
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import src.config as config
from src.modeling import build_pipeline
from src.evaluation import log_production_plots

def main():
    """Main execution block combining data ingestion, tuning, logging, and model registry."""
    print("Loading data...")
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    
    X = train_df.drop(columns=['Depression', 'id'])
    y = train_df['Depression']
    test_X = test_df.drop(columns=['id'])
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=21
    )

    # --- DATA VERSIONING: Hash the dataset to detect silent data changes ---
    train_data_hash = hashlib.md5(pd.util.hash_pandas_object(train_df).values).hexdigest()

    # --- 1. MLFLOW CONFIGURATION ---
    # MLFLOW: Defines the local database to permanently store experiment metadata
    mlflow.set_tracking_uri("sqlite:///mlflow.db") 
    
    # MLFLOW: Creates or connects to a "folder" to organize all runs for this project
    mlflow.set_experiment("Depression_Prediction_Optuna")

    # --- 2. OPTUNA OBJECTIVE ---
    def objective(trial):
        """Optuna objective function to test hyperparameters using Nested MLflow Runs."""
        
        # MLFLOW: 'nested=True' creates a child run under the main script run
        with mlflow.start_run(nested=True):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }
            
            # MLFLOW: Logs the exact hyperparameters tested in this specific trial
            mlflow.log_params(params)
            
            # Build and train
            pipeline = build_pipeline(xgb_params=params)
            pipeline.fit(X_train, y_train)
            
            preds = pipeline.predict(X_valid)
            score = f1_score(y_valid, preds)
            
            # MLFLOW: Logs the performance metric so you can sort trials in the UI
            mlflow.log_metric("val_f1_score", score)
            return score

    # --- 3. EXECUTE OPTUNA TUNING ---
    # --- RUN VERSIONING: Dynamically append timestamp to guarantee a unique run name ---
    run_name = f"XGBoost_Optimization_{time.strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Starting Optuna Tuning: {run_name}")
    
    # MLFLOW: Starts the main "Parent" run to track the overall execution
    with mlflow.start_run(run_name=run_name):
        
        # MLFLOW: Tags make it easy to filter and search for runs later
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("stage", "experimentation")
        
        # MLFLOW (Data Versioning): Log the dataset size and cryptographic hash
        mlflow.log_param("train_data_hash", train_data_hash)
        mlflow.log_param("train_rows", len(X_train))
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        
        print(f"\nBest Trial F1: {study.best_value}")
        
        # MLFLOW: Explicitly save the winning parameters to the parent run summary
        mlflow.log_params({"best_" + k: v for k, v in study.best_params.items()})

        # --- 4. TRAIN & REGISTER FINAL MODEL ---
        print("\nTraining final model with best parameters...")
        final_pipeline = build_pipeline(xgb_params=study.best_params)
        final_pipeline.fit(X_train, y_train)
        
        final_preds = final_pipeline.predict(X_valid)
        print("\nFinal Classification Report:")
        print(classification_report(y_valid, final_preds))

        final_f1 = f1_score(y_valid, final_preds) 
        
        # MLFLOW: Log the final decisive metric
        mlflow.log_metric("final_test_f1", final_f1)

        # --- 5. PRODUCTION EVALUATION & EXPLAINABILITY ---
        print("Generating Production-Level Visualizations...")
        log_production_plots(final_pipeline, X_valid, y_valid)

        # --- MODEL SIGNATURE (To prevent Schema Mismatches) ---
        # MLFLOW: Infers the exact data types and shape the model expects
        signature = infer_signature(X_train, final_preds)

        # MLFLOW: Packages the model, logs its dependencies, and pushes it to the Model Registry
        mlflow.sklearn.log_model(
            sk_model=final_pipeline, 
            artifact_path="model",
            registered_model_name="Depression_Classifier_Final",
            signature=signature  # Enforces data contract
        )
        
        # --- 6. GENERATE SUBMISSION ---
        print("Generating predictions for test set...")
        test_preds = final_pipeline.predict(test_X)
        submission = pd.DataFrame({'id': test_df['id'], 'Depression': test_preds})
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

if __name__ == "__main__":
    main()