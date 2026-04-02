import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import src.config as config
from src.modeling import build_pipeline

def main():
    print("Loading data...")
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    
    # Separate features and target
    X = train_df.drop(columns=['Depression', 'id'])
    y = train_df['Depression']
    test_X = test_df.drop(columns=['id'])
    
    # Validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=21
    )
    
    print("Building and training pipeline...")
    pipeline = build_pipeline()
    
    # Fit the pipeline on the training set
    pipeline.fit(X_train, y_train)
    
    print("Evaluating on validation set...")
    val_preds = pipeline.predict(X_valid)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_valid, val_preds))
    print("\nClassification Report:")
    print(classification_report(y_valid, val_preds))
    
    print("Generating predictions for test set...")
    # Because we packaged everything into a scikit-learn pipeline, 
    # generating predictions is a single, clean function call.
    test_preds = pipeline.predict(test_X)
    
    submission = pd.DataFrame({
        'id': test_df['id'],
        'Depression': test_preds
    })
    
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")

if __name__ == "__main__":
    main()