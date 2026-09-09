"""
Feature Engineering Module for Customer Churn Prediction.
Handles data cleaning, derived feature engineering, and encoding/scaling pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Strips whitespace
    - Converts TotalCharges to numeric
    - Imputes missing TotalCharges
    - Drops customerID
    """
    df = df.copy()

    # Drop customer ID if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Clean TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # Impute missing values with MonthlyCharges * tenure
        missing_mask = df["TotalCharges"].isna()
        if missing_mask.sum() > 0:
            df.loc[missing_mask, "TotalCharges"] = df.loc[missing_mask, "MonthlyCharges"] * df.loc[missing_mask, "tenure"]

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates informative business features:
    1. TenureCohort: Group tenure into customer lifecycle stages.
    2. ServiceCount: Total number of active internet add-on services subscribed.
    3. HasSecurityBundle: Both OnlineSecurity and TechSupport active.
    4. HasStreamingBundle: Both StreamingTV and StreamingMovies active.
    5. MonthlyToTotalRatio: MonthlyCharges / (TotalCharges + 1).
    6. HighRiskContractFiber: Month-to-month contract with Fiber optic and no tech support.
    """
    df = df.copy()

    # 1. Tenure Cohorts
    tenure = df["tenure"] if "tenure" in df.columns else 0
    df["TenureCohort"] = pd.cut(
        tenure,
        bins=[-1, 12, 24, 48, 73],
        labels=["New_0-12m", "Developing_13-24m", "Established_25-48m", "Loyal_49-72m"],
    )

    # 2. Add-on service count
    addon_services = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    service_count = 0
    for col in addon_services:
        if col in df.columns:
            service_count = service_count + (df[col] == "Yes").astype(int)
    df["ServiceCount"] = service_count

    # 3. Bundles
    if "OnlineSecurity" in df.columns and "TechSupport" in df.columns:
        df["HasSecurityBundle"] = ((df["OnlineSecurity"] == "Yes") & (df["TechSupport"] == "Yes")).astype(int)
    else:
        df["HasSecurityBundle"] = 0

    if "StreamingTV" in df.columns and "StreamingMovies" in df.columns:
        df["HasStreamingBundle"] = ((df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes")).astype(int)
    else:
        df["HasStreamingBundle"] = 0

    # 4. Monthly to Total Charges Ratio
    if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
        df["MonthlyToTotalRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1.0)
    else:
        df["MonthlyToTotalRatio"] = 0.0

    # 5. High Risk Profile
    if "Contract" in df.columns and "InternetService" in df.columns and "TechSupport" in df.columns:
        df["HighRiskContractFiber"] = (
            (df["Contract"] == "Month-to-month")
            & (df["InternetService"] == "Fiber optic")
            & (df["TechSupport"] != "Yes")
        ).astype(int)
    else:
        df["HighRiskContractFiber"] = 0

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer for numerical scaling and categorical encoding.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_data(df: pd.DataFrame):
    """
    Full preparation pipeline:
    - Cleans raw data
    - Feature engineering
    - Splits into features X and target y
    """
    df_clean = clean_raw_data(df)
    df_engineered = engineer_features(df_clean)

    if "Churn" in df_engineered.columns:
        y = (df_engineered["Churn"] == "Yes").astype(int)
        X = df_engineered.drop(columns=["Churn"])
    else:
        y = None
        X = df_engineered

    return X, y, df_engineered
