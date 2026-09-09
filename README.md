# 🚀 Customer Churn Prediction & Retention Engine
### *An End-to-End Production-Grade Machine Learning Pipeline*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1185FE.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Table of Contents
- [1. Business Problem & Project Overview](#-1-business-problem--project-overview)
- [2. System Architecture](#-2-system-architecture)
- [3. Dataset & Data Dictionary](#-3-dataset--data-dictionary)
- [4. Advanced Feature Engineering](#-4-advanced-feature-engineering)
- [5. Machine Learning Models](#-5-machine-learning-models)
- [6. Performance & Evaluation Metrics](#-6-performance--evaluation-metrics)
- [7. Visual Reports & Confusion Matrices](#-7-visual-reports--confusion-matrices)
- [8. Interactive Web Dashboard](#-8-interactive-web-dashboard)
- [9. Project Structure](#-9-project-structure)
- [10. Quickstart Guide (How to Run)](#-10-quickstart-guide-how-to-run)
- [11. API Documentation](#-11-api-documentation)
- [12. Key Takeaways for Technical Interviews](#-12-key-takeaways-for-technical-interviews)

---

## 💼 1. Business Problem & Project Overview

Customer churn (the percentage of customers who discontinue a subscription or service) is one of the most critical challenges facing the subscription and telecommunications industry. 

* **The Problem**: Acquiring a new customer costs **5x to 25x more** than retaining an existing customer.
* **The Goal**: Build a predictive machine learning system that accurately identifies high-risk customers before they cancel, allowing retention and customer success teams to take proactive, cost-effective measures.
* **The Solution**: 
  - An automated pipeline comparing **4 classification algorithms** (Logistic Regression, Decision Tree, Random Forest, XGBoost).
  - Business-driven **Feature Engineering** that captures customer stickiness, lifecycle stages, and risk profiles.
  - Evaluation focused on **Precision, Recall, F1-score, and ROC-AUC** rather than misleading raw accuracy.
  - A responsive **FastAPI web application** that provides real-time churn probability scoring and actionable retention playbooks.

---

## 🏗️ 2. System Architecture

```
                                  [ Raw Input Data ]
                                          │
                                          ▼
                             [ Data Cleaning & Imputation ]
                         (Missing TotalCharges, whitespace strip)
                                          │
                                          ▼
                            [ Domain Feature Engineering ]
                    (Tenure Cohorts, Service Density, Risk Flags)
                                          │
                                          ▼
                       [ Scikit-Learn ColumnTransformer Pipeline ]
                        ├── StandardScaler (Numerical features)
                        └── OneHotEncoder(drop='first') (Categoricals)
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
         [ 80% Train Split ]                             [ 20% Test Split ]
                  │                                               │
                  ▼                                               │
      [ Multi-Model Training ]                                    │
       ├── Logistic Regression                                    │
       ├── Decision Tree                                          │
       ├── Random Forest                                          │
       └── XGBoost (scale_pos_weight)                             │
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                           [ Evaluation & Model Selection ]
                     (Precision, Recall, F1, Confusion Matrices, ROC-AUC)
                                          │
                                          ▼
                           [ Production Model Persistence ]
                        (best_model.joblib & preprocessor.joblib)
                                          │
                                          ▼
                         [ FastAPI Web & Inference Dashboard ]
                   (Real-time scoring, Risk gauge, Retention advice)
```

---

## 📊 3. Dataset & Data Dictionary

The project uses the benchmark **Telco Customer Churn** schema with 4,000 customer records and 21 attributes:

| Feature Name | Type | Description | Values / Range |
| :--- | :---: | :--- | :--- |
| `customerID` | Categorical | Unique customer identifier | e.g. `CUST-00001` (dropped for modeling) |
| `gender` | Categorical | Customer gender | `Male`, `Female` |
| `SeniorCitizen` | Binary | Whether customer is senior citizen | `0` (No), `1` (Yes) |
| `Partner` | Binary | Customer has spouse/partner | `Yes`, `No` |
| `Dependents` | Binary | Customer has dependents | `Yes`, `No` |
| `tenure` | Numeric | Months the customer has been with company | `1` to `72` |
| `PhoneService` | Binary | Subscribed to telephone service | `Yes`, `No` |
| `MultipleLines` | Categorical | Multiple phone lines | `No phone service`, `No`, `Yes` |
| `InternetService` | Categorical | Type of internet provision | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity` | Categorical | Internet security add-on | `Yes`, `No`, `No internet service` |
| `OnlineBackup` | Categorical | Cloud backup add-on | `Yes`, `No`, `No internet service` |
| `DeviceProtection`| Categorical | Hardware insurance plan | `Yes`, `No`, `No internet service` |
| `TechSupport` | Categorical | Dedicated technical support | `Yes`, `No`, `No internet service` |
| `StreamingTV` | Categorical | TV streaming add-on | `Yes`, `No`, `No internet service` |
| `StreamingMovies` | Categorical | Movie streaming add-on | `Yes`, `No`, `No internet service` |
| `Contract` | Categorical | Billing contract frequency | `Month-to-month`, `One year`, `Two year` |
| `PaperlessBilling`| Binary | Digital invoice delivery | `Yes`, `No` |
| `PaymentMethod` | Categorical | Payment transaction channel | `Electronic check`, `Credit card`, etc. |
| `MonthlyCharges` | Numeric | Monthly bill amount in USD | `\$18.25` - `\$118.75` |
| `TotalCharges` | Numeric | Cumulative charges paid in USD | `\$18.25` - `\$8,684.00` |
| `Churn` **[TARGET]** | Binary | Whether customer canceled service | `Yes` (1), `No` (0) |

---

## 💡 4. Advanced Feature Engineering

Feature engineering is what separates standard academic models from production-grade machine learning. We engineered 6 new business-driven features:

1. **`TenureCohort` (Customer Lifecycle Stage)**:
   - Categorizes tenure into 4 strategic buckets:
     - `New_0-12m`: Highest vulnerability window.
     - `Developing_13-24m`: Transitioning period.
     - `Established_25-48m`: Solid retention.
     - `Loyal_49-72m`: Highly engaged advocates.
2. **`ServiceCount` (Customer Stickiness / Switching Cost)**:
   - Sum of subscribed digital add-on services (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`). Customers with higher service counts face high friction when switching providers, leading to lower churn.
3. **`HasSecurityBundle`**:
   - Binary flag (`1` if both `OnlineSecurity` and `TechSupport` are active).
4. **`HasStreamingBundle`**:
   - Binary flag (`1` if both `StreamingTV` and `StreamingMovies` are active).
5. **`MonthlyToTotalRatio`**:
   - `MonthlyCharges / (TotalCharges + 1.0)`. Highlights new customers carrying disproportionately high monthly bills early in their lifecycle.
6. **`HighRiskContractFiber` (Key Interaction Feature)**:
   - Flag detecting customers with `Month-to-month` contract AND `Fiber optic` internet AND `No Tech Support`. This segment statistically represents the highest churn probability.

---

## 🤖 5. Machine Learning Models

We trained and tuned 4 complementary classification models:

1. **Logistic Regression (Baseline Linear Classifier)**
   - Configured with L2 penalty, `solver='lbfgs'`, `max_iter=1000`, and `class_weight='balanced'`.
   - Highly interpretable baseline with well-calibrated probabilities.
2. **Decision Tree Classifier**
   - Configured with `max_depth=6`, `min_samples_split=20`, and `min_samples_leaf=10` to avoid tree over-expansion and overfitting.
   - Provides clear decision rules for stakeholders.
3. **Random Forest Classifier (Bagging Ensemble)**
   - Configured with 150 independent decision trees, `max_depth=8`, `min_samples_leaf=8`, and `class_weight='balanced'`.
   - Drastically reduces variance and provides robust feature importances.
4. **XGBoost Classifier (Gradient Boosting - Selected Best Model 🏆)**
   - Configured with 150 boosting rounds, `learning_rate=0.05`, `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8`, and `scale_pos_weight` set dynamically based on class distribution.
   - Incrementally learns from errors of previous trees, providing the highest predictive power on tabular data.

---

## 📈 6. Performance & Evaluation Metrics

### Why Not Just Accuracy?
In churn prediction, datasets often suffer from class imbalance. A dummy model predicting "No Churn" for every customer could achieve 75% accuracy while completely failing the business. We evaluated models across 5 holistic metrics:

* **Precision** ($\frac{TP}{TP + FP}$): Minimizes false alarms (prevents wasting retention discounts on loyal customers).
* **Recall / Sensitivity** ($\frac{TP}{TP + FN}$): Minimizes missed churners (catches customers before they leave).
* **F1-Score** ($2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$): Harmonic mean balancing Precision and Recall.
* **ROC-AUC**: Evaluates discrimination ability across all classification probability thresholds.

### 🏆 Test Set Benchmark Results (800 Unseen Customers)

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **75.75%** | **0.7204** | **0.7486** | **0.7342** | **0.8302** | 🏆 **BEST MODEL** |
| **Decision Tree** | 75.00% | 0.7005 | 0.7709 | 0.7340 | 0.8184 | Strong Baseline |
| **Logistic Regression** | 74.75% | 0.7063 | 0.7458 | 0.7255 | 0.8327 | Excellent Calibration |
| **Random Forest** | 73.62% | 0.6889 | 0.7486 | 0.7175 | 0.8209 | High Stability |

### 🎯 Detailed Confusion Matrix Breakdown (XGBoost)
* **True Negatives (TN)**: `338` (Correctly identified loyal customers who stayed)
* **True Positives (TP)**: `268` (Correctly identified churners who were at risk)
* **False Positives (FP)**: `104` (False alarms: predicted to churn, but stayed)
* **False Negatives (FN)**: `90` (Missed churners: predicted to stay, but churned)

---

## 🖼️ 7. Visual Reports & Confusion Matrices

All visual reports are automatically generated and saved during pipeline execution:

| Report Type | Saved Path | Description |
| :--- | :--- | :--- |
| **Confusion Matrices** | `reports/confusion_matrices.png` | 2x2 comparison grid showing TP, FP, TN, FN for all 4 models |
| **ROC Curves** | `reports/roc_curves.png` | Receiver Operating Characteristic curves with AUC scores |
| **Model Comparison** | `reports/model_comparison.png` | Grouped bar chart comparing Accuracy, Precision, Recall, F1, ROC |
| **Feature Importance** | `reports/feature_importance.png` | Top 15 churn drivers determined by XGBoost |

---

## 💻 8. Interactive Web Dashboard

The project includes an interactive web application built with **FastAPI** and modern vanilla frontend styling:

* **Benchmark Explorer**: Live table displaying the performance of all 4 models.
* **Interactive Risk Calculator**: Dropdown menus and numeric sliders for Tenure, Monthly Charges, Contract type, Payment method, and Add-ons.
* **Instant Churn Gauge**: Shows real-time Churn Probability percentage and Color-coded Risk Tier (**LOW RISK**, **MODERATE RISK**, **CRITICAL RISK**).
* **Automated Retention Playbook**: Generates custom business recommendations (e.g. *"Offer 1-year contract incentive"*, *"Bundle complimentary VIP tech support"*).

---

## 📁 9. Project Structure

```
Customer Churn Prediction/
├── data/
│   ├── raw_churn_data.csv            # Benchmark / raw customer dataset
│   └── processed_churn_data.csv      # Cleaned & feature-engineered dataset
├── models/
│   ├── best_model.joblib             # Serialized best classifier (XGBoost)
│   ├── preprocessor.joblib           # Fitted ColumnTransformer (Scaler + Encoder)
│   ├── all_models.joblib             # Dictionary of all 4 fitted models
│   └── metrics_summary.json          # Machine-readable evaluation metrics
├── reports/
│   ├── confusion_matrices.png        # 4-Model Confusion Matrix grid
│   ├── roc_curves.png                # ROC-AUC curves comparison plot
│   ├── model_comparison.png          # Side-by-side metric comparison bar chart
│   └── feature_importance.png        # Top 15 predictive features
├── src/
│   ├── __init__.py
│   ├── dataset.py                    # Data loader & benchmark generator
│   ├── feature_engineering.py        # Cleaning, derived features, ColumnTransformer
│   ├── models.py                     # Logistic Regression, DT, RF, XGBoost setups
│   ├── evaluate.py                   # Metrics calculation & visualization suite
│   ├── pipeline.py                   # Automated end-to-end training orchestrator
│   └── predict.py                    # Inference engine & retention recommendation logic
├── app.py                            # Interactive FastAPI web application
├── main.py                           # One-click CLI entrypoint
├── requirements.txt                  # Pinned production dependencies
└── README.md                         # Complete project documentation
```

---

## 🚀 10. Quickstart Guide (How to Run)

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
```

### Step 2: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 3: Run Training & Evaluation Pipeline
```bash
python main.py
```
*Outputs: Trains all 4 models, prints comparison table, computes confusion matrices, saves plots to `reports/`, and serializes models to `models/`.*

### Step 4: Launch Web Dashboard
```bash
python app.py
```
Open your browser at:
👉 **`http://127.0.0.1:8000`**

---

## 🔌 11. API Documentation

The FastAPI application provides two REST API endpoints for external integration:

### 1. Get Model Benchmarks
* **Endpoint**: `GET /api/metrics`
* **Response**: JSON containing test set scores (Accuracy, Precision, Recall, F1, ROC-AUC) for all 4 models.

### 2. Predict Customer Churn
* **Endpoint**: `POST /api/predict`
* **Request Payload**:
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 3,
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
  "TotalCharges": 268.50
}
```
* **Response Payload**:
```json
{
  "churn_prediction": "Churn (Will Leave)",
  "churn_probability": 0.967,
  "churn_probability_percentage": "96.7%",
  "risk_tier": "CRITICAL / HIGH RISK",
  "risk_color": "#ef4444",
  "recommendations": [
    "Offer a discount incentive to switch from Month-to-Month to a 1-year or 2-year contract.",
    "Bundle complimentary VIP Tech Support & security onboarding.",
    "Review customer usage patterns and offer an optimized loyalty bundle."
  ]
}
```

---

## 🧠 12. Key Takeaways for Technical Interviews

When presenting this project in an interview, emphasize these 4 technical decisions:

1. **Strict Data Leakage Prevention**: We fit `StandardScaler` and `OneHotEncoder` strictly on the 80% train split and applied `transform` to the test split.
2. **Business-Informed Feature Engineering**: Rather than feeding raw columns directly into a model, we engineered `ServiceCount` and `HighRiskContractFiber` which emerged as the top non-linear churn drivers.
3. **Handling Class Imbalance**: We avoided synthetic noise from SMOTE on tabular categories by utilizing cost-sensitive learning (`class_weight='balanced'` and XGBoost `scale_pos_weight`).
4. **Full-Stack ML Engineering**: Went beyond modeling in a notebook to create modular code (`src/`), automated testing pipelines (`main.py`), and a user-facing REST API & Dashboard (`app.py`).

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
