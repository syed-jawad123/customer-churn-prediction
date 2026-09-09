"""
Prediction & Inference Module for Customer Churn Prediction.
Loads saved preprocessor and model to provide churn probability and retention recommendations.
"""

import os
import joblib
import pandas as pd
from src.feature_engineering import clean_raw_data, engineer_features


class ChurnPredictor:
    def __init__(self, model_path: str = "models/best_model.joblib", preprocessor_path: str = "models/preprocessor.joblib"):
        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            raise FileNotFoundError("Model or preprocessor artifacts not found. Please run main.py first!")
        
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict_customer(self, customer_data: dict) -> dict:
        """
        Takes a customer dictionary, applies feature engineering and preprocessing,
        and outputs churn probability, classification, risk level, and recommendations.
        """
        df = pd.DataFrame([customer_data])
        df_clean = clean_raw_data(df)
        df_engineered = engineer_features(df_clean)

        # Preprocess features
        X_trans = self.preprocessor.transform(df_engineered)

        churn_pred = int(self.model.predict(X_trans)[0])
        churn_proba = float(self.model.predict_proba(X_trans)[0][1]) if hasattr(self.model, "predict_proba") else (1.0 if churn_pred == 1 else 0.0)

        # Risk assessment
        if churn_proba >= 0.70:
            risk_tier = "CRITICAL / HIGH RISK"
            color = "#ef4444"
        elif churn_proba >= 0.40:
            risk_tier = "MODERATE RISK"
            color = "#f59e0b"
        else:
            risk_tier = "LOW RISK"
            color = "#10b981"

        # Actionable business retention strategies
        recommendations = []
        if customer_data.get("Contract") == "Month-to-month":
            recommendations.append("Offer a discount incentive to switch from Month-to-Month to a 1-year or 2-year contract.")
        if customer_data.get("TechSupport") != "Yes":
            recommendations.append("Bundle complimentary VIP Tech Support & security onboarding.")
        if float(customer_data.get("MonthlyCharges", 0)) > 75:
            recommendations.append("Review customer usage patterns and offer an optimized loyalty bundle.")
        if customer_data.get("PaymentMethod") == "Electronic check":
            recommendations.append("Encourage automated bank or credit card payment with a recurring bill credit.")
        if not recommendations:
            recommendations.append("Customer shows strong retention signals. Maintain standard relationship management.")

        return {
            "churn_prediction": "Churn (Will Leave)" if churn_pred == 1 else "Loyal (Will Stay)",
            "churn_probability": round(churn_proba, 4),
            "churn_probability_percentage": f"{churn_proba * 100:.1f}%",
            "risk_tier": risk_tier,
            "risk_color": color,
            "recommendations": recommendations,
        }
