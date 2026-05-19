"""
Column detection and mapping for uploaded churn CSVs (Telco schema as reference).
"""

from __future__ import annotations

import re

import pandas as pd

from src.preprocess import load_feature_config

# Aliases → canonical telco column name
COLUMN_ALIASES: dict[str, list[str]] = {
    "customerID": ["customerid", "customer_id", "id", "cust_id", "account_id"],
    "Churn": ["churn", "is_churn", "churned", "target", "exited"],
    "tenure": ["tenure", "months_active", "customer_tenure"],
    "MonthlyCharges": ["monthlycharges", "monthly_charges", "monthly_fee"],
    "TotalCharges": ["totalcharges", "total_charges", "lifetime_value"],
    "Contract": ["contract", "contract_type"],
    "InternetService": ["internetservice", "internet_service", "internet"],
    "gender": ["gender", "sex"],
    "SeniorCitizen": ["seniorcitizen", "senior_citizen", "is_senior"],
    "Partner": ["partner", "has_partner"],
    "Dependents": ["dependents", "has_dependents"],
    "PhoneService": ["phoneservice", "phone_service"],
    "MultipleLines": ["multiplelines", "multiple_lines"],
    "PaperlessBilling": ["paperlessbilling", "paperless_billing"],
    "PaymentMethod": ["paymentmethod", "payment_method"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def detect_column_mapping(uploaded_columns: list[str]) -> dict[str, str | None]:
    """
    Map canonical telco names → uploaded column name (or None if missing).
    """
    config = load_feature_config()
    required = (
        [config["id_column"], config["target_column"]]
        + config["numeric_features"]
        + config["categorical_features"]
    )
    # Engineered cols may be created later
    skip_engineered = {"ServiceCount", "HasInternet", "IsMonthToMonth", "AvgChargePerTenure"}
    required = [c for c in required if c not in skip_engineered]

    norm_uploaded = {_norm(c): c for c in uploaded_columns}
    mapping: dict[str, str | None] = {}

    for canonical in required:
        if canonical in uploaded_columns:
            mapping[canonical] = canonical
            continue
        found = None
        for alias in COLUMN_ALIASES.get(canonical, [_norm(canonical)]):
            if alias in norm_uploaded:
                found = norm_uploaded[alias]
                break
        if found is None:
            for col in uploaded_columns:
                if _norm(canonical) in _norm(col) or _norm(col) in _norm(canonical):
                    found = col
                    break
        mapping[canonical] = found

    return mapping


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Rename uploaded columns to canonical telco names."""
    rename = {uploaded: canonical for canonical, uploaded in mapping.items() if uploaded}
    return df.rename(columns=rename)


def validate_for_pipeline(
    df: pd.DataFrame,
    require_target: bool = False,
) -> tuple[bool, list[str]]:
    """Return (ok, list of missing canonical columns) after mapping."""
    config = load_feature_config()
    raw_required = (
        ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService"]
        + config["categorical_features"]
    )
    if require_target:
        raw_required.append(config["target_column"])
    raw_required = list(dict.fromkeys(raw_required))
    missing = [c for c in raw_required if c not in df.columns]
    return len(missing) == 0, missing


def is_telco_compatible(df: pd.DataFrame) -> bool:
    """True if upload already uses IBM telco column names (minus engineered)."""
    ok, _ = validate_for_pipeline(df)
    return ok
