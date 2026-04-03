import os
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
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

    # --- 1. MLFLOW CONFIGURATION ---
    mlflow.set_tracking_uri("sqlite:///mlflow.db") 
    mlflow.set_experiment("Depression_Prediction_Optuna")

    # --- 2. OPTUNA OBJECTIVE ---
    def objective(trial):
        with mlflow.start_run(nested=True):
            # FIXED: Removed 'random_state' from here because build_pipeline 
            # already adds it manually.
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 7),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
            }
            
            mlflow.log_params(params)
            
            # Build and train
            pipeline = build_pipeline(xgb_params=params)
            pipeline.fit(X_train, y_train)
            
            preds = pipeline.predict(X_valid)
            score = f1_score(y_valid, preds)
            
            mlflow.log_metric("val_f1_score", score)
            return score

    # --- 3. EXECUTE OPTUNA TUNING ---
    # Tip: Increment the v-number if you change features or data
    run_name = "XGBoost_Optimization_v1" 
    
    print(f"Starting Optuna Tuning: {run_name}")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("stage", "experimentation")
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        
        print(f"\nBest Trial F1: {study.best_value}")
        mlflow.log_params({"best_" + k: v for k, v in study.best_params.items()})

        # --- 4. TRAIN & REGISTER FINAL MODEL ---
        print("\nTraining final model with best parameters...")
        final_pipeline = build_pipeline(xgb_params=study.best_params)
        final_pipeline.fit(X_train, y_train)
        
        final_preds = final_pipeline.predict(X_valid)
        print("\nFinal Classification Report:")
        print(classification_report(y_valid, final_preds))

        final_f1 = f1_score(y_valid, final_preds) # Calculate the winner's score
        mlflow.log_metric("final_test_f1", final_f1)

        # This automatically creates/updates the model in the "Models" tab
        mlflow.sklearn.log_model(
            sk_model=final_pipeline, 
            artifact_path="model",
            registered_model_name="Depression_Classifier_Final"
        )
        
        # --- 5. GENERATE SUBMISSION ---
        print("Generating predictions for test set...")
        test_preds = final_pipeline.predict(test_X)
        submission = pd.DataFrame({'id': test_df['id'], 'Depression': test_preds})
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

if __name__ == "__main__":
    main()