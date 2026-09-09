"""
Training & Evaluation Pipeline Module for Customer Churn Prediction.
Orchestrates data loading, feature engineering, model training, evaluation, and artifact persistence.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.dataset import load_dataset
from src.feature_engineering import prepare_data, build_preprocessor
from src.models import get_models
from src.evaluate import (
    evaluate_single_model,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_metrics_comparison,
    plot_feature_importance,
)


def run_pipeline(data_path: str = "data/raw_churn_data.csv", test_size: float = 0.20, random_state: int = 42) -> dict:
    """
    Executes the full end-to-end churn prediction pipeline.
    """
    print("\n========================================================")
    print("      CUSTOMER CHURN PREDICTION PIPELINE START          ")
    print("========================================================\n")

    # 1. Load Data
    raw_df = load_dataset(data_path)
    print(f"[*] Dataset shape: {raw_df.shape}")
    print(f"[*] Churn distribution:\n{raw_df['Churn'].value_counts(normalize=True).to_dict()}\n")

    # 2. Feature Engineering & Preparation
    print("[*] Performing Feature Engineering...")
    X, y, df_engineered = prepare_data(raw_df)

    # Save processed data
    os.makedirs("data", exist_ok=True)
    df_engineered.to_csv("data/processed_churn_data.csv", index=False)
    print(f"[+] Engineered features: {list(X.columns)}")
    print(f"[+] Saved processed data to data/processed_churn_data.csv")

    # 3. Train/Test Split (Stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[*] Train set size: {X_train_raw.shape[0]} | Test set size: {X_test_raw.shape[0]}")

    # 4. Fit Preprocessor (Scaler + One-Hot Encoder)
    print("[*] Fitting Preprocessor (Scaling & OneHotEncoding)...")
    preprocessor = build_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Extract feature names after One-Hot Encoding
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"feat_{i}" for i in range(X_train.shape[1])]
    print(f"[+] Encoded Feature Vector Dimension: {X_train.shape[1]}")

    # 5. Initialize Models
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    models = get_models(scale_pos_weight=pos_weight)

    results = {}
    fitted_models = {}

    print("\n[*] Training & Evaluating Models:")
    print("-" * 75)
    print(f"{'Model Name':<22} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1 Score':<9} | {'ROC-AUC':<9}")
    print("-" * 75)

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Evaluate
        metrics = evaluate_single_model(model, X_test, y_test, name)
        results[name] = metrics

        print(
            f"{name:<22} | {metrics['accuracy']:<9.4f} | {metrics['precision']:<9.4f} | "
            f"{metrics['recall']:<9.4f} | {metrics['f1_score']:<9.4f} | {metrics['roc_auc']:<9.4f}"
        )

    print("-" * 75)

    # 6. Model Selection (Highest F1 Score)
    best_model_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = fitted_models[best_model_name]
    print(f"\n[+] BEST MODEL SELECTED: '{best_model_name}' with F1-Score: {results[best_model_name]['f1_score']:.4f}")

    # 7. Generate Evaluation Reports & Plots
    print("\n[*] Generating Visual Evaluation Reports in reports/ ...")
    os.makedirs("reports", exist_ok=True)
    plot_confusion_matrices(results, output_path="reports/confusion_matrices.png")
    plot_roc_curves(results, y_test, output_path="reports/roc_curves.png")
    plot_metrics_comparison(results, output_path="reports/model_comparison.png")

    # Plot feature importance for best model or Random Forest/XGBoost
    if "XGBoost" in fitted_models:
        plot_feature_importance(fitted_models["XGBoost"], feature_names, "XGBoost", "reports/feature_importance.png")
    elif "Random Forest" in fitted_models:
        plot_feature_importance(fitted_models["Random Forest"], feature_names, "Random Forest", "reports/feature_importance.png")

    # 8. Save Models & Preprocessor
    print("\n[*] Saving Trained Model Artifacts to models/ ...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    joblib.dump(fitted_models, "models/all_models.joblib")

    # Clean results for JSON serialization (remove large arrays)
    json_summary = {
        "best_model": best_model_name,
        "feature_names": feature_names,
        "test_sample_count": len(y_test),
        "models": {},
    }
    for name, res in results.items():
        json_summary["models"][name] = {
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1_score": res["f1_score"],
            "roc_auc": res["roc_auc"],
            "confusion_matrix": res["confusion_matrix"],
        }

    with open("models/metrics_summary.json", "w") as f:
        json.dump(json_summary, f, indent=4)
    print("[+] Saved metrics summary to models/metrics_summary.json")

    print("\n========================================================")
    print("      CUSTOMER CHURN PREDICTION PIPELINE COMPLETED       ")
    print("========================================================\n")

    return {
        "results": results,
        "best_model_name": best_model_name,
        "feature_names": feature_names,
        "metrics_summary": json_summary,
    }
