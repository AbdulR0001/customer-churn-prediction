from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    "tenure_months", "monthly_charge", "support_calls", "late_payments",
    "contract_type", "internet_service", "paperless_billing", "senior_citizen"
]
TARGET = "churn"


def generate_customer_data(n_samples=3000, random_state=42):
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, n_samples)
    monthly = np.clip(rng.normal(85, 28, n_samples), 20, 180).round(2)
    support = np.clip(rng.poisson(1.7, n_samples), 0, 10)
    late = np.clip(rng.poisson(0.8, n_samples), 0, 6)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.27, 0.18])
    internet = rng.choice(["Fiber optic", "DSL", "None"], n_samples, p=[0.52, 0.38, 0.10])
    paperless = rng.choice(["Yes", "No"], n_samples, p=[0.67, 0.33])
    senior = rng.choice([0, 1], n_samples, p=[0.82, 0.18])

    score = (-1.7 - 0.027 * tenure + 0.014 * (monthly - 70) + 0.28 * support +
             0.38 * late + 1.1 * (contract == "Month-to-month") -
             0.65 * (contract == "Two year") + 0.45 * (internet == "Fiber optic") +
             0.25 * (paperless == "Yes") + 0.35 * senior)
    probability = 1 / (1 + np.exp(-score))
    churn = rng.binomial(1, probability)
    return pd.DataFrame({
        "tenure_months": tenure, "monthly_charge": monthly,
        "support_calls": support, "late_payments": late,
        "contract_type": contract, "internet_service": internet,
        "paperless_billing": paperless, "senior_citizen": senior, "churn": churn
    })


def load_or_create_data(path="data/customer_churn.csv"):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        generate_customer_data().to_csv(path, index=False)
    return pd.read_csv(path)
