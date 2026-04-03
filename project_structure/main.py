import os
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import src.config as config
from src.modeling import build_pipeline

def main():
    print("Loading data...")
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    
    X = train_df.drop(columns=['Depression', 'id'])
    y = train_df['Depression']
    test_X = test_df.drop(columns=['id'])
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=21
    )

    # --- SETUP MLFLOW ---
    # In a real company, this URI would point to a remote server. Here, it creates a local folder.
    mlflow.set_tracking_uri("sqlite:///mlflow.db") 
    mlflow.set_experiment("Depression_Prediction_Optuna")

    # --- SETUP OPTUNA OBJECTIVE ---
    def objective(trial):
        with mlflow.start_run(nested=True):
            # 1. Define hyperparameters to search
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }
            
            # 2. Log parameters to MLflow
            mlflow.log_params(params)
            
            # 3. Build and train pipeline
            pipeline = build_pipeline(xgb_params=params)
            pipeline.fit(X_train, y_train)
            
            # 4. Predict and evaluate
            preds = pipeline.predict(X_valid)
            score = f1_score(y_valid, preds) # Optimizing for F1-score
            
            # 5. Log metric to MLflow
            mlflow.log_metric("val_f1_score", score)
            
            return score

    print("Starting Optuna Hyperparameter Tuning (10 trials)...")
    # Start a parent MLflow run to group the Optuna trials
    with mlflow.start_run(run_name="Optuna_Optimization"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        
        print("\nBest Trial:")
        print(f"  F1 Score: {study.best_value}")
        print(f"  Params: {study.best_params}")
        
        # Log best params to the parent run
        mlflow.log_params({"best_" + k: v for k, v in study.best_params.items()})

        # --- TRAIN FINAL MODEL WITH BEST PARAMS ---
        print("\nTraining final model with best parameters...")
        final_pipeline = build_pipeline(xgb_params=study.best_params)
        final_pipeline.fit(X_train, y_train)
        
        # Evaluate Final Model
        final_preds = final_pipeline.predict(X_valid)
        print("\nFinal Classification Report:")
        print(classification_report(y_valid, final_preds))
        
        # Log the final scikit-learn pipeline artifact to MLflow
        mlflow.sklearn.log_model(final_pipeline, "best_pipeline_model")
        
        # Generate Test Predictions
        print("Generating predictions for test set...")
        test_preds = final_pipeline.predict(test_X)
        submission = pd.DataFrame({'id': test_df['id'], 'Depression': test_preds})
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

if __name__ == "__main__":
    main()