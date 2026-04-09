import matplotlib.pyplot as plt
import scikitplot as skplt
import shap
import mlflow
import os

def log_production_plots(model_pipeline, X_valid, y_valid, run_path="plots"):
    """
    Generates and logs professional evaluation plots to MLflow.
    """
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    # 1. ROC-AUC and Confusion Matrix
    y_probas = model_pipeline.predict_proba(X_valid)
    
    # Plot ROC
    skplt.metrics.plot_roc(y_valid, y_probas)
    plt.savefig(f"{run_path}/roc_curve.png")
    plt.close()

    # Plot Confusion Matrix
    skplt.metrics.plot_confusion_matrix(y_valid, model_pipeline.predict(X_valid), normalize=True)
    plt.savefig(f"{run_path}/confusion_matrix.png")
    plt.close()

    # 2. SHAP Explainability (The 'Why')
    # We extract the model and the transformed data
    model = model_pipeline.named_steps['classifier']
    transformed_data = model_pipeline.named_steps['encoding'].transform(
        model_pipeline.named_steps['feature_engineering'].transform(X_valid)
    )
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_data)

    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, transformed_data, show=False)
    plt.savefig(f"{run_path}/shap_summary.png")
    plt.close()

    # 3. Log everything to MLflow
    mlflow.log_artifacts(run_path)
    print(f"Production plots logged to MLflow artifacts.")