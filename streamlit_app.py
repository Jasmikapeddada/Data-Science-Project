"""
Customer Segmentation & Churn Retention Analytics Platform
COA Network Pvt. Limited — Data Science Internship Project

Run locally:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_registry import (
    DEFAULT_LABEL,
    assess_upload,
    delete_upload,
    get_option_by_label,
    list_dataset_options,
    load_dataset_raw,
    save_acceptable_upload,
)
from src.predict import load_models, log_scoring_run, score_dataframe
from src.schema import detect_column_mapping

st.set_page_config(
    page_title="Churn & Segmentation Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_NEW_LABEL = "➕ Upload new CSV..."

_SKIP = frozenset(
    {"ServiceCount", "HasInternet", "IsMonthToMonth", "AvgChargePerTenure", "Churn", "customerID"}
)


@st.cache_resource
def _load_models():
    return load_models()


@st.cache_data(show_spinner="Scoring customers…")
def _score_cached(dataset_id: str, _path_token: str, mapping_json: str) -> pd.DataFrame:
    """Cache scores per saved dataset (path token busts cache when file changes)."""
    opt = next((o for o in list_dataset_options() if o["id"] == dataset_id), None)
    if opt is None:
        raise ValueError(f"Unknown dataset id: {dataset_id}")

    df = load_dataset_raw(opt)
    if opt["kind"] == "upload":
        return score_dataframe(df, column_mapping=None)
    mapping = json.loads(mapping_json) if mapping_json else None
    return score_dataframe(df, column_mapping=mapping)


def render_sidebar_mapping(uploaded: pd.DataFrame) -> dict[str, str | None] | None:
    """Column mapping UI when auto-detect cannot match all required fields."""
    from src.schema import validate_for_pipeline

    auto = detect_column_mapping(list(uploaded.columns))
    missing = [k for k, v in auto.items() if v is None and k not in _SKIP]

    if not missing:
        return auto

    st.sidebar.warning("Map your columns to the telco schema below.")
    final: dict[str, str | None] = {}
    options = ["— skip —"] + list(uploaded.columns)
    with st.sidebar.expander("Column mapping", expanded=True):
        for canonical, detected in auto.items():
            if canonical in _SKIP:
                continue
            default_idx = options.index(detected) if detected in options else 0
            choice = st.selectbox(canonical, options, index=default_idx, key=f"map_{canonical}")
            final[canonical] = None if choice == "— skip —" else choice

    renamed = uploaded.rename(columns={v: k for k, v in final.items() if v})
    ok, still_missing = validate_for_pipeline(renamed)
    if not ok:
        st.sidebar.error(f"Still missing: {', '.join(still_missing)}")
        return None
    return final


def _dataset_select_labels() -> list[str]:
    return [o["label"] for o in list_dataset_options()] + [UPLOAD_NEW_LABEL]


def _render_upload_flow() -> None:
    st.info(
        "Upload a **customer churn** CSV with telco-style fields (tenure, charges, contract, "
        "services, etc.). If it passes validation, it is saved under `data/uploads/` and "
        "added to the dataset list."
    )
    file = st.sidebar.file_uploader("Customer churn CSV", type=["csv"], key="new_upload_file")
    if file is None:
        st.markdown(
            """
**Expected columns (examples):** tenure, MonthlyCharges, Contract, InternetService,  
gender, PhoneService, … — see `data/DATA_DICTIONARY.md`.
            """
        )
        return

    uploaded = pd.read_csv(file)
    st.sidebar.metric("Uploaded rows", f"{len(uploaded):,}")

    mapping = render_sidebar_mapping(uploaded)
    if mapping is None:
        return

    ok, msg, missing = assess_upload(uploaded, mapping)
    if not ok:
        st.error(msg)
        if missing:
            st.caption(f"Missing columns: {', '.join(missing)}")
        st.warning(
            "This model expects **telecom-style customer churn** data. "
            "Bank-only datasets will not pass unless columns are aligned to the telco schema."
        )
        return

    if st.sidebar.button("Validate & save dataset", type="primary"):
        try:
            entry = save_acceptable_upload(uploaded, file.name, mapping)
            _score_cached.clear()
            st.session_state["dataset_select"] = entry["label"]
            st.success(
                f"Saved **{entry['label']}** ({entry['row_count']:,} rows) → `{entry['saved_path']}`"
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def _render_saved_datasets_manager() -> None:
    uploads = [o for o in list_dataset_options() if o["kind"] == "upload"]
    if not uploads:
        return
    with st.sidebar.expander("Saved uploads", expanded=False):
        for opt in uploads:
            col1, col2 = st.columns([3, 1])
            col1.caption(opt["label"])
            if col2.button("Del", key=f"del_{opt['id']}", help="Remove saved file"):
                delete_upload(opt["id"])
                _score_cached.clear()
                if st.session_state.get("dataset_select") == opt["label"]:
                    st.session_state["dataset_select"] = DEFAULT_LABEL
                st.rerun()


def main() -> None:
    _load_models()

    st.title("Customer Segmentation & Churn Analytics")
    st.caption(
        "COA Network Pvt. Limited · Internship project · "
        "Telco-trained models · Saved compatible uploads in `data/uploads/`"
    )

    st.sidebar.header("Data source")
    labels = _dataset_select_labels()
    if "dataset_select" not in st.session_state:
        st.session_state["dataset_select"] = DEFAULT_LABEL

    selected_label = st.sidebar.selectbox(
        "Choose dataset",
        labels,
        key="dataset_select",
    )

    _render_saved_datasets_manager()

    if selected_label == UPLOAD_NEW_LABEL:
        _render_upload_flow()
        return

    opt = get_option_by_label(selected_label)
    if opt is None:
        st.error("Dataset not found.")
        return

    path_token = opt["path"] or "default"
    mapping_json = json.dumps(opt.get("column_mapping") or {})
    try:
        scored = _score_cached(opt["id"], path_token, mapping_json)
        log_scoring_run(scored, source=opt["id"])
    except Exception as exc:
        st.error(f"Could not load or score dataset: {exc}")
        return

    if opt["kind"] == "upload":
        st.caption(f"Using saved dataset: **{opt['label']}** (`{opt['path']}`)")
    else:
        st.caption("Using default **Telco Customer Churn** benchmark (7,043 customers).")

    # ---------------------------------------------------------------------------
    # KPI row
    # ---------------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(scored):,}")
    if "Churn_numeric" in scored.columns:
        c2.metric("Churn rate", f"{scored['Churn_numeric'].mean() * 100:.1f}%")
    else:
        c2.metric("Churn rate", "N/A")
    c3.metric("High risk", f"{(scored['risk_band'] == 'High').sum():,}")
    c4.metric("Avg churn prob.", f"{scored['churn_probability'].mean():.2f}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Segments", "Churn risk", "Download"]
    )

    with tab1:
        st.subheader("Risk distribution")
        fig = px.histogram(
            scored,
            x="churn_probability",
            color="risk_band",
            nbins=40,
            title="Predicted churn probability",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

        if "Churn_numeric" in scored.columns:
            st.subheader("Actual vs predicted (sample)")
            st.dataframe(
                scored[
                    ["customerID", "Churn", "churn_probability", "risk_band", "segment_name"]
                ].head(20),
                use_container_width=True,
            )

    with tab2:
        st.subheader("Customer segments")
        seg = (
            scored.groupby(["Segment", "segment_name"], as_index=False)
            .agg(
                customers=("customerID", "count"),
                avg_prob=("churn_probability", "mean"),
                churn_rate=("Churn_numeric", "mean")
                if "Churn_numeric" in scored.columns
                else ("churn_probability", "mean"),
            )
        )
        if "Churn_numeric" in scored.columns:
            seg["churn_rate"] = (seg["churn_rate"] * 100).round(1)
        st.dataframe(seg, use_container_width=True)

        fig2 = px.scatter(
            scored.sample(min(1500, len(scored)), random_state=42),
            x="tenure",
            y="MonthlyCharges",
            color="segment_name",
            opacity=0.6,
            title="Tenure vs monthly charges by segment",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("High-risk customers")
        high = scored[scored["risk_band"] == "High"].sort_values(
            "churn_probability", ascending=False
        )
        st.write(f"**{len(high):,}** customers in High risk band")
        st.dataframe(
            high[
                [
                    "customerID",
                    "segment_name",
                    "churn_probability",
                    "risk_band",
                    "tenure",
                    "Contract",
                    "MonthlyCharges",
                ]
            ].head(50),
            use_container_width=True,
        )

        band_summary = scored.groupby("risk_band").agg(
            n=("customerID", "count"),
            avg_prob=("churn_probability", "mean"),
        )
        if "Churn_numeric" in scored.columns:
            band_summary["actual_churn"] = scored.groupby("risk_band")["Churn_numeric"].mean()
        st.dataframe(band_summary, use_container_width=True)

    with tab4:
        st.subheader("Export scored customers")
        export_cols = [
            "customerID",
            "Segment",
            "segment_name",
            "churn_probability",
            "risk_band",
            "churn_predicted",
            "tenure",
            "MonthlyCharges",
            "Contract",
        ]
        if "Churn" in scored.columns:
            export_cols.insert(1, "Churn")
        export_df = scored[[c for c in export_cols if c in scored.columns]]
        st.download_button(
            "Download CSV",
            export_df.to_csv(index=False).encode("utf-8"),
            file_name="customers_scored.csv",
            mime="text/csv",
        )
        st.caption("Use this file for retention campaigns.")

    st.sidebar.divider()
    st.sidebar.markdown("**Monitoring**")
    log_path = ROOT / "monitoring" / "logs" / "prediction_runs.csv"
    if log_path.exists():
        logs = pd.read_csv(log_path).tail(5)
        st.sidebar.dataframe(logs, hide_index=True)
    else:
        st.sidebar.caption("Prediction logs will appear after first run.")


if __name__ == "__main__":
    main()
