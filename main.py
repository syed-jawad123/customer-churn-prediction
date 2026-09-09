"""
Main Entrypoint for Customer Churn Prediction Project.
Executes the training pipeline, model comparison, evaluation plots, and sample inference.
"""

import sys
import os
from src.pipeline import run_pipeline
from src.predict import ChurnPredictor


def main():
    print("=" * 65)
    print("    AI-POWERED CUSTOMER CHURN PREDICTION SYSTEM    ")
    print("    Models: Logistic Regression | Decision Tree    ")
    print("            Random Forest       | XGBoost          ")
    print("=" * 65)

    # 1. Run Pipeline
    pipeline_result = run_pipeline(data_path="data/raw_churn_data.csv")
    best_model_name = pipeline_result["best_model_name"]
    results = pipeline_result["results"]

    print("\n[+] Pipeline execution completed successfully.")
    print(f"[+] Best Model: {best_model_name}")

    # 2. Print Detailed Confusion Matrix breakdown for each model
    print("\n" + "=" * 65)
    print("      CONFUSION MATRIX & DETAILED METRICS BREAKDOWN        ")
    print("=" * 65)

    for name, res in results.items():
        cm = res["confusion_matrix"]
        print(f"\n--- {name} ---")
        print(f"  Accuracy : {res['accuracy'] * 100:.2f}%")
        print(f"  Precision: {res['precision']:.4f}  (Ability to avoid false positives)")
        print(f"  Recall   : {res['recall']:.4f}     (Ability to detect all churners)")
        print(f"  F1 Score : {res['f1_score']:.4f}   (Harmonic balance)")
        print(f"  ROC-AUC  : {res['roc_auc']:.4f}")
        print("  Confusion Matrix:")
        print(f"    - True Negatives  (Predicted Stay, Actually Stayed) : {cm['tn']}")
        print(f"    - False Positives (Predicted Churn, Actually Stayed): {cm['fp']}")
        print(f"    - False Negatives (Predicted Stay, Actually Churned): {cm['fn']}")
        print(f"    - True Positives  (Predicted Churn, Actually Churned): {cm['tp']}")

    # 3. Test Live Inference with ChurnPredictor
    print("\n" + "=" * 65)
    print("            SAMPLE CUSTOMER REAL-TIME INFERENCE            ")
    print("=" * 65)

    predictor = ChurnPredictor()

    sample_customers = [
        {
            "description": "High-Risk Customer (Month-to-month, Fiber, no tech support, high bill)",
            "data": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.50,
                "TotalCharges": 179.0,
            },
        },
        {
            "description": "Low-Risk Customer (Two-year contract, high tenure, auto-pay, bundled)",
            "data": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "Yes",
                "tenure": 60,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Two year",
                "PaperlessBilling": "No",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 64.00,
                "TotalCharges": 3840.0,
            },
        },
    ]

    for sample in sample_customers:
        pred = predictor.predict_customer(sample["data"])
        print(f"\n[Test Profile]: {sample['description']}")
        print(f"  -> Prediction : {pred['churn_prediction']}")
        print(f"  -> Churn Risk : {pred['churn_probability_percentage']} ({pred['risk_tier']})")
        print(f"  -> Actions    : {'; '.join(pred['recommendations'])}")

    print("\n" + "=" * 65)
    print("  Artifacts generated:")
    print("    - Model:    models/best_model.joblib")
    print("    - Scaler:   models/preprocessor.joblib")
    print("    - Reports:  reports/confusion_matrices.png")
    print("                reports/roc_curves.png")
    print("                reports/model_comparison.png")
    print("                reports/feature_importance.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
