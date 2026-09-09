"""
Dataset module for Customer Churn Prediction.
Loads existing CSV dataset or generates a realistic benchmark Telco dataset.
"""

import os
import numpy as np
import pandas as pd


def generate_benchmark_churn_data(n_samples: int = 4000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a statistically realistic customer churn dataset based on Telco benchmark schema.
    """
    np.random.seed(random_state)

    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_samples + 1)]
    genders = np.random.choice(["Male", "Female"], size=n_samples)
    senior_citizens = np.random.choice([0, 1], size=n_samples, p=[0.83, 0.17])
    partners = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.where(
        partners == "Yes",
        np.random.choice(["Yes", "No"], size=n_samples, p=[0.55, 0.45]),
        np.random.choice(["Yes", "No"], size=n_samples, p=[0.10, 0.90]),
    )

    # Tenure between 1 and 72 months (skewed towards new customers and long-term loyal)
    tenure_raw = np.random.beta(0.8, 0.8, size=n_samples)
    tenure = np.clip(np.round(tenure_raw * 71 + 1), 1, 72).astype(int)

    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["No", "Yes"], p=[0.58, 0.42]))

    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22])

    online_sec, online_backup, dev_protect, tech_supp, stream_tv, stream_movies = [], [], [], [], [], []
    for net in internet_service:
        if net == "No":
            for lst in [online_sec, online_backup, dev_protect, tech_supp, stream_tv, stream_movies]:
                lst.append("No internet service")
        else:
            online_sec.append(np.random.choice(["No", "Yes"], p=[0.60, 0.40]))
            online_backup.append(np.random.choice(["No", "Yes"], p=[0.55, 0.45]))
            dev_protect.append(np.random.choice(["No", "Yes"], p=[0.56, 0.44]))
            tech_supp.append(np.random.choice(["No", "Yes"], p=[0.62, 0.38]))
            stream_tv.append(np.random.choice(["No", "Yes"], p=[0.50, 0.50]))
            stream_movies.append(np.random.choice(["No", "Yes"], p=[0.49, 0.51]))

    # Contract selection depends somewhat on tenure
    contracts = []
    for t in tenure:
        if t < 12:
            contracts.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.82, 0.14, 0.04]))
        elif t < 36:
            contracts.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.50, 0.35, 0.15]))
        else:
            contracts.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.25, 0.35, 0.40]))

    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    payment_methods = np.random.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=n_samples,
        p=[0.35, 0.22, 0.22, 0.21],
    )

    # Monthly charges calculation based on selected services
    monthly_charges = []
    for i in range(n_samples):
        charge = 20.0
        if phone_service[i] == "Yes":
            charge += 15.0
        if multiple_lines[i] == "Yes":
            charge += 10.0
        if internet_service[i] == "DSL":
            charge += 25.0
        elif internet_service[i] == "Fiber optic":
            charge += 45.0
        if online_sec[i] == "Yes":
            charge += 6.0
        if online_backup[i] == "Yes":
            charge += 6.0
        if dev_protect[i] == "Yes":
            charge += 6.0
        if tech_supp[i] == "Yes":
            charge += 8.0
        if stream_tv[i] == "Yes":
            charge += 12.0
        if stream_movies[i] == "Yes":
            charge += 12.0
        # Add a little jitter
        charge += np.random.normal(0, 1.5)
        monthly_charges.append(round(max(18.5, charge), 2))

    monthly_charges = np.array(monthly_charges)
    total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 15, size=n_samples), 2)
    total_charges = np.maximum(total_charges, monthly_charges)

    # Convert a few TotalCharges to blank string / missing to emulate realistic real-world dirty data
    total_charges_str = [str(tc) for tc in total_charges]
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.003), replace=False)
    for idx in missing_indices:
        total_charges_str[idx] = " "

    # Realistic Churn Probability calculation (log-odds)
    # High risk: month-to-month, low tenure, high charges, fiber without tech support, electronic check
    z = -1.2
    for i in range(n_samples):
        zi = z
        if contracts[i] == "Month-to-month":
            zi += 1.4
        elif contracts[i] == "Two year":
            zi -= 1.6

        if tenure[i] <= 12:
            zi += 0.9
        elif tenure[i] >= 48:
            zi -= 1.1

        if internet_service[i] == "Fiber optic":
            zi += 0.6
            if tech_supp[i] == "No":
                zi += 0.5
        elif internet_service[i] == "No":
            zi -= 0.8

        if payment_methods[i] == "Electronic check":
            zi += 0.4
        if paperless_billing[i] == "Yes":
            zi += 0.3
        if senior_citizens[i] == 1:
            zi += 0.2
        if monthly_charges[i] > 80:
            zi += 0.35

        prob = 1.0 / (1.0 + np.exp(-zi))
        churn_prob = np.clip(prob, 0.02, 0.95)

    churn_flags = []
    for i in range(n_samples):
        zi = z
        if contracts[i] == "Month-to-month":
            zi += 1.4
        elif contracts[i] == "Two year":
            zi -= 1.6

        if tenure[i] <= 12:
            zi += 0.9
        elif tenure[i] >= 48:
            zi -= 1.1

        if internet_service[i] == "Fiber optic":
            zi += 0.6
            if tech_supp[i] == "No":
                zi += 0.5
        elif internet_service[i] == "No":
            zi -= 0.8

        if payment_methods[i] == "Electronic check":
            zi += 0.4
        if paperless_billing[i] == "Yes":
            zi += 0.3
        if senior_citizens[i] == 1:
            zi += 0.2
        if monthly_charges[i] > 80:
            zi += 0.35

        prob = 1.0 / (1.0 + np.exp(-zi))
        churn_flags.append("Yes" if np.random.rand() < prob else "No")

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": genders,
            "SeniorCitizen": senior_citizens,
            "Partner": partners,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": dev_protect,
            "TechSupport": tech_supp,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_movies,
            "Contract": contracts,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_methods,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges_str,
            "Churn": churn_flags,
        }
    )

    return df


def load_dataset(file_path: str = "data/raw_churn_data.csv") -> pd.DataFrame:
    """
    Loads dataset from file_path. If file doesn't exist, generates benchmark data,
    saves it to file_path, and returns it.
    """
    if os.path.exists(file_path):
        print(f"[*] Loading dataset from {file_path}")
        df = pd.read_csv(file_path)
    else:
        print(f"[!] {file_path} not found. Generating benchmark Telco churn dataset...")
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        df = generate_benchmark_churn_data(n_samples=4000)
        df.to_csv(file_path, index=False)
        print(f"[+] Saved generated benchmark dataset to {file_path} (Shape: {df.shape})")

    return df
