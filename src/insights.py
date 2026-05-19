"""
Day 5 — Business insights, retention priorities, and report artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocess import PROCESSED_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROCESSED_DIR / "figures"

# Action playbook: segment_name → recommended action
SEGMENT_ACTIONS = {
    "Loyal Long-Tenure": "Upsell premium bundles; referral rewards; protect with loyalty perks",
    "Month-to-Month Volatile": "Offer annual contract discount; assign retention call within 7 days",
    "Standard Mixed": "Low-touch email nurture; monitor usage; avoid over-discounting",
    "At-Risk New Customers": "Welcome call + onboarding support; first-90-day check-in",
    "High-Spend Stable": "VIP support tier; exclusive offers; early renewal incentives",
    "Growing Accounts": "Education on service bundles; tenure-based loyalty milestones",
    "Critical Churn Risk": "Immediate outreach; win-back offer; escalate to account manager",
}


def load_scored() -> pd.DataFrame:
    path = PROCESSED_DIR / "telco_scored.csv"
    if not path.exists():
        raise FileNotFoundError("Run Day 4 first: python scripts/run_train_churn.py")
    return pd.read_csv(path)


def load_cluster_profiles() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "cluster_profiles.csv")


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Simple markdown table without tabulate dependency."""
    headers = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.values
    ]
    return "\n".join([headers, sep] + rows)


def load_churn_metrics() -> dict:
    with open(MODELS_DIR / "churn_metrics.json", encoding="utf-8") as f:
        return json.load(f)


def build_segment_risk_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab segment × risk with churn rates and recommended actions."""
    summary = (
        scored.groupby(["segment_name", "risk_band"], as_index=False)
        .agg(
            customers=("customerID", "count"),
            avg_churn_probability=("churn_probability", "mean"),
            actual_churn_rate=("Churn_numeric", "mean"),
        )
    )
    summary["actual_churn_rate_pct"] = (summary["actual_churn_rate"] * 100).round(1)
    summary["avg_churn_probability"] = summary["avg_churn_probability"].round(3)
    summary["recommended_action"] = summary["segment_name"].map(SEGMENT_ACTIONS).fillna(
        "Review account individually"
    )
    summary["priority_score"] = (
        summary["avg_churn_probability"] * summary["customers"]
    ).round(1)
    return summary.sort_values("priority_score", ascending=False)


def build_retention_priority_list(
    scored: pd.DataFrame,
    top_n: int = 500,
) -> pd.DataFrame:
    """Rank customers for proactive retention (high prob + valuable segments)."""
    df = scored.copy()
    # Prioritize high risk; among ties prefer higher monthly value proxy via segment
    high_risk = df[df["risk_band"] == "High"].copy()
    if len(high_risk) < top_n:
        medium = df[df["risk_band"] == "Medium"].nlargest(top_n - len(high_risk), "churn_probability")
        priority = pd.concat([high_risk, medium], ignore_index=True)
    else:
        priority = high_risk.nlargest(top_n, "churn_probability")

    cols = [
        "customerID",
        "segment_name",
        "risk_band",
        "churn_probability",
        "Churn",
        "Churn_numeric",
    ]
    cols = [c for c in cols if c in priority.columns]
    out = priority[cols].head(top_n).reset_index(drop=True)
    out["priority_rank"] = out.index + 1
    out["recommended_action"] = out["segment_name"].map(SEGMENT_ACTIONS).fillna(
        "Retention outreach"
    )
    return out


def build_executive_kpis(scored: pd.DataFrame, metrics: dict) -> dict:
    champion = metrics["champion"]
    m = metrics.get(champion) or metrics["logistic_regression"]

    return {
        "total_customers": int(len(scored)),
        "overall_churn_rate_pct": round(scored["Churn_numeric"].mean() * 100, 2),
        "high_risk_customers": int((scored["risk_band"] == "High").sum()),
        "high_risk_pct_of_base": round(
            (scored["risk_band"] == "High").mean() * 100, 1
        ),
        "champion_model": champion,
        "model_recall": round(m["recall"], 3),
        "model_roc_auc": round(m["roc_auc"], 3),
        "churners_captured_in_top_20pct": round(m["churners_in_top_pct"] * 100, 1),
        "segments": int(scored["segment_name"].nunique()),
    }


def plot_segment_risk_heatmap(summary: pd.DataFrame, out_path: Path) -> Path:
    pivot = summary.pivot(
        index="segment_name",
        columns="risk_band",
        values="actual_churn_rate_pct",
    )
    col_order = [c for c in ["Low", "Medium", "High"] if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "Churn %"})
    ax.set_title("Actual churn rate (%) by segment and risk band")
    ax.set_xlabel("Risk band")
    ax.set_ylabel("Segment")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_retention_funnel(scored: pd.DataFrame, out_path: Path) -> Path:
    order = ["Low", "Medium", "High"]
    counts = scored["risk_band"].value_counts().reindex(order).fillna(0)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    ax.bar(counts.index, counts.values, color=colors, edgecolor="black")
    ax.set_ylabel("Customers")
    ax.set_xlabel("Risk band")
    ax.set_title("Customer base by churn risk tier")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, f"{int(v):,}", ha="center")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_markdown_report(
    kpis: dict,
    profiles: pd.DataFrame,
    summary: pd.DataFrame,
    scored: pd.DataFrame,
    top_insights: list[str],
) -> str:
    """Build internship insights document body."""
    summary_top = _df_to_markdown_table(
        summary.head(8)[
            [
                "segment_name",
                "risk_band",
                "customers",
                "actual_churn_rate_pct",
                "recommended_action",
            ]
        ]
    )

    insights_bullets = "\n".join(f"- {x}" for x in top_insights)

    return f"""# Customer Segmentation & Churn — Insights & Recommendations

**Project:** COA Network Pvt. Limited — Data Science Internship  
**Dataset:** IBM Telco Customer Churn (7,043 customers)  
**Models:** K-Means segmentation (K=4) + Logistic Regression churn classifier

---

## 1. Executive summary

{insights_bullets}

---

## 2. Key performance indicators

| KPI | Value |
|-----|-------|
| Total customers analyzed | {kpis['total_customers']:,} |
| Overall churn rate | {kpis['overall_churn_rate_pct']}% |
| High-risk customers (model) | {kpis['high_risk_customers']:,} ({kpis['high_risk_pct_of_base']}% of base) |
| Champion churn model | {kpis['champion_model']} |
| Model recall (test set) | {kpis['model_recall']} |
| ROC-AUC | {kpis['model_roc_auc']} |
| Churners captured in top 20% risk scores | {kpis['churners_captured_in_top_20pct']}% |
| Customer segments | {kpis['segments']} |

---

## 3. Segmentation insights

| Segment | Customers | Churn % | Avg tenure (mo) | Avg monthly $ | Top contract |
|---------|-----------|---------|-----------------|---------------|--------------|
{chr(10).join(
    f"| {r.segment_name} | {int(r.customers):,} | {r.churn_rate_pct:.1f} | {r.tenure:.1f} | {r.MonthlyCharges:.1f} | {r.top_contract} |"
    for _, r in profiles.iterrows()
)}

### Segment interpretation

1. **Loyal Long-Tenure** (~2,000 customers, ~12% churn)  
   Long tenure, higher spend, mostly two-year contracts. **Strategy:** retention is strong — focus on upsell and referrals, not discounts.

2. **Month-to-Month Volatile** (segments 0 & 3, ~3,500 customers, 41–51% churn)  
   Short-to-medium tenure, month-to-month contracts, elevated churn. **Strategy:** highest priority for contract migration campaigns and proactive calls.

3. **Standard Mixed** (~1,540 customers, ~7% churn)  
   Lower monthly charges, minimal services. **Strategy:** low-touch nurture; avoid costly interventions unless risk score rises.

---

## 4. Churn risk tiers

| Risk band | Customers | Avg predicted probability | Actual churn rate |
|-----------|-----------|---------------------------|-------------------|
{chr(10).join(_risk_table_rows(scored))}

**Action rule:** Focus retention budget on **High** band first, then **Medium** where segment is Month-to-Month Volatile.

---

## 5. Top priority matrix (segment × risk)

{summary_top}

---

## 6. Recommended actions (playbook)

| Segment | Risk | Action |
|---------|------|--------|
| Month-to-Month Volatile | High | Call within 7 days; offer 15–20% annual plan discount |
| Month-to-Month Volatile | Medium | SMS + email contract upgrade reminder |
| Loyal Long-Tenure | High | VIP check-in (billing dispute / service issue likely) |
| Loyal Long-Tenure | Low | Referral program + add-on bundle offer |
| Standard Mixed | Low | Automated satisfaction survey only |

Full priority list: `data/processed/retention_priority_list.csv` (top 500 accounts).

---

## 7. Business impact (estimated narrative for report)

- Targeting **top 20% risk** (~1,400 customers) captures ~**half** of eventual churners with focused outreach.
- Shifting **10%** of month-to-month volatile segment to annual contracts could reduce churn materially (industry benchmark).
- **Cost saving:** Retention outreach on scored list vs. blanket campaigns reduces wasted contact on low-risk loyal base (~46% in Low band).

---

## 8. Limitations & ethics

- Public benchmark data — not COA Network live CRM.
- Model trained on historical patterns; retrain when customer behavior drifts.
- Avoid discriminatory targeting using sensitive attributes; focus on behavior (tenure, contract, spend).

---

## 9. Next steps (Days 6–7)

- Deploy Streamlit dashboard with upload + scoring
- Add prediction logging and drift monitoring
- Present: live demo on high-risk segment + retention CSV download

---

*Generated by Day 5 insights pipeline.*
"""


def _risk_table_rows(scored: pd.DataFrame) -> list[str]:
    rows = []
    for band in ["Low", "Medium", "High"]:
        sub = scored[scored["risk_band"] == band]
        if len(sub) == 0:
            continue
        rows.append(
            f"| {band} | {len(sub):,} | {sub['churn_probability'].mean():.3f} | "
            f"{sub['Churn_numeric'].mean() * 100:.1f}% |"
        )
    return rows


def compute_auto_insights(
    scored: pd.DataFrame,
    profiles: pd.DataFrame,
    metrics: dict,
) -> list[str]:
    m = metrics["logistic_regression"]
    worst = profiles.loc[profiles["churn_rate_pct"].idxmax()]
    best = profiles.loc[profiles["churn_rate_pct"].idxmin()]
    high_n = (scored["risk_band"] == "High").sum()

    return [
        f"Overall churn rate is **{scored['Churn_numeric'].mean() * 100:.1f}%** across {len(scored):,} customers.",
        f"Highest-risk segment is **{worst['segment_name']}** ({worst['churn_rate_pct']:.1f}% churn, {int(worst['customers']):,} customers).",
        f"Most stable segment is **{best['segment_name']}** ({best['churn_rate_pct']:.1f}% churn).",
        f"**{high_n:,}** customers ({high_n / len(scored) * 100:.0f}%) are in the **High** risk band and need proactive retention.",
        f"Champion model (**{metrics['champion']}**) achieves **{m['recall']:.0%} recall** and **{m['roc_auc']:.2f} ROC-AUC** on held-out test data.",
        f"Concentrating on the top 20% risk scores captures **{m['churners_in_top_pct']:.0%}** of actual churners — efficient targeting for marketing spend.",
    ]


def run_insights(top_n: int = 500) -> dict:
    scored = load_scored()
    profiles = load_cluster_profiles()
    metrics = load_churn_metrics()

    kpis = build_executive_kpis(scored, metrics)
    summary = build_segment_risk_summary(scored)
    priority = build_retention_priority_list(scored, top_n=top_n)
    insights = compute_auto_insights(scored, profiles, metrics)
    report_md = generate_markdown_report(kpis, profiles, summary, scored, insights)

    return {
        "scored": scored,
        "profiles": profiles,
        "metrics": metrics,
        "kpis": kpis,
        "summary": summary,
        "priority": priority,
        "insights": insights,
        "report_md": report_md,
    }


def save_insights_artifacts(result: dict) -> dict[str, Path]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    report_path = DOCS_DIR / "INSIGHTS_AND_RECOMMENDATIONS.md"
    report_path.write_text(result["report_md"], encoding="utf-8")
    paths["insights_report"] = report_path

    summary_path = PROCESSED_DIR / "segment_risk_summary.csv"
    result["summary"].to_csv(summary_path, index=False)
    paths["segment_risk_summary"] = summary_path

    priority_path = PROCESSED_DIR / "retention_priority_list.csv"
    result["priority"].to_csv(priority_path, index=False)
    paths["retention_priority_list"] = priority_path

    kpi_path = PROCESSED_DIR / "executive_kpis.json"
    with open(kpi_path, "w", encoding="utf-8") as f:
        json.dump(result["kpis"], f, indent=2)
    paths["executive_kpis"] = kpi_path

    paths["segment_risk_heatmap"] = plot_segment_risk_heatmap(
        result["summary"],
        FIGURES_DIR / "segment_risk_heatmap.png",
    )
    paths["retention_funnel"] = plot_retention_funnel(
        result["scored"],
        FIGURES_DIR / "retention_risk_funnel.png",
    )

    return paths
