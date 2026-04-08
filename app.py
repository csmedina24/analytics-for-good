"""
Bringing SF Back — Policy Analysis Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Bringing SF Back", layout="wide")
sns.set_theme(style="whitegrid", font_scale=1.05)
BLUE = "#2563EB"
RED = "#DC2626"
GREEN = "#16A34A"


# ── DATA + MODELS ─────────────────────────────
@st.cache_data
def load_and_fit():
    crime = pd.read_csv("data/crime_panel.csv")
    housing = pd.read_csv("data/housing_panel.csv")

    c = crime.dropna(subset=["crime_rate", "density", "log_income"])
    h = housing.dropna(subset=["log_income", "crime_rate"])

    m_total = smf.ols(
        "crime_rate ~ density + log_median_value + pct_residential "
        "+ log_income + in_hospitality_zone + C(year)", data=c
    ).fit(cov_type="HC1")

    m_violent = smf.ols(
        "violent_rate ~ density + log_median_value + pct_residential "
        "+ log_income + in_hospitality_zone + C(year)", data=c
    ).fit(cov_type="HC1")

    m_housing = smf.ols(
        "log_median_value ~ density + pct_residential + building_age "
        "+ median_stories + crime_rate + violent_rate + log_income + C(year)",
        data=h
    ).fit(cov_type="HC1")

    return crime, housing, m_total, m_violent, m_housing

crime_panel, housing_panel, m_total, m_violent, m_housing = load_and_fit()


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def coef_chart(model, keep_vars, labels, title, color):
    coefs = model.params[keep_vars]
    ci = model.conf_int().loc[keep_vars]
    errs = np.array([coefs - ci[0], ci[1] - coefs])
    fig, ax = plt.subplots(figsize=(7, max(2.8, len(keep_vars) * 0.55)))
    ax.barh(range(len(keep_vars)), coefs.values, xerr=errs, height=0.5,
            color=color, alpha=0.85, edgecolor="white", capsize=4,
            error_kw={"linewidth": 1.5, "color": "#374151"})
    ax.axvline(0, color="#6B7280", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_yticks(range(len(keep_vars)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title(title, fontweight="bold", fontsize=13)
    for i, var in enumerate(keep_vars):
        star = sig_stars(model.pvalues[var])
        if star:
            x_pos = coefs.values[i] + errs[1][i] + abs(coefs.values).max() * 0.03
            ax.text(x_pos, i, star, va="center", fontsize=11,
                    fontweight="bold", color="#DC2626")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# ── PAGE ──────────────────────────────────────
st.title("Bringing SF Back")

tab1, tab2, tab3, tab4 = st.tabs([
    "Crime Regression", "Housing Regression", "Policy Review", "Data + Variables"
])


# ── CRIME ─────────────────────────────────────
with tab1:
    keep = ["density", "log_median_value", "pct_residential",
            "log_income", "in_hospitality_zone"]
    labels = ["Housing Density", "Property Value (log)",
              "% Residential Land", "Household Income (log)",
              "In Hospitality Zone"]

    col1, col2 = st.columns(2)
    with col1:
        fig = coef_chart(m_total, keep, labels,
                         f"Total Crime Rate  |  R² = {m_total.rsquared:.3f}", BLUE)
        st.pyplot(fig)
        plt.close()
    with col2:
        fig = coef_chart(m_violent, keep, labels,
                         f"Violent Crime Rate  |  R² = {m_violent.rsquared:.3f}", RED)
        st.pyplot(fig)
        plt.close()


# ── HOUSING ───────────────────────────────────
with tab2:
    keep_h = ["density", "pct_residential", "building_age",
              "median_stories", "crime_rate", "violent_rate", "log_income"]
    labels_h = ["Housing Density", "% Residential Land",
                "Building Age (years)", "Building Height (stories)",
                "Total Crime Rate", "Violent Crime Rate",
                "Household Income (log)"]

    fig = coef_chart(m_housing, keep_h, labels_h,
                     f"Property Value Drivers  |  R² = {m_housing.rsquared:.3f}", GREEN)
    st.pyplot(fig)
    plt.close()


# ── POLICY REVIEW ─────────────────────────────
with tab3:
    st.markdown("""
    | Policy | Housing | Safety | Transit |
    |--------|---------|--------|---------|
    | **Upzoning** (Temescal, Oakland) | Values may rise with density | Property crime up, violent unchanged | More riders on existing routes |
    | **Office-to-Residential** (100 Van Ness) | Surrounding values grew slower | Residents exposed to high-crime area | Ridership up, equity concern N vs S of Market |
    | **Hospitality Task Force** (Feb 2026) | Safety perception affects values | May reduce visible crime, risk displacement | N/A |
    """)


# ── DATA + VARIABLES ──────────────────────────
with tab4:
    col_data, col_dict = st.columns([1.3, 1])

    with col_data:
        display = crime_panel[["analysis_neighborhood", "year", "crime_rate",
                                "violent_rate", "property_rate", "density",
                                "median_hh_income", "pct_residential",
                                "in_hospitality_zone"]].copy()
        display.columns = ["Neighborhood", "Year", "Total Crime Rate",
                            "Violent Crime Rate", "Property Crime Rate",
                            "Housing Density", "Median HH Income",
                            "% Residential", "Hospitality Zone"]
        st.dataframe(display.round(2), use_container_width=True, hide_index=True)

    with col_dict:
        st.markdown("""
        | Variable | Definition |
        |----------|-----------|
        | **Total Crime Rate** | All crimes per 1,000 residents |
        | **Violent Crime Rate** | Assault, robbery, homicide per 1,000 |
        | **Property Crime Rate** | Theft, burglary, vandalism per 1,000 |
        | **Housing Density** | Avg housing units per parcel |
        | **Median HH Income** | Median household income ($) |
        | **% Residential** | Share of parcels zoned residential (0-1) |
        | **Hospitality Zone** | 1 = in downtown patrol zone |
        | **Property Value (log)** | Log of median assessed value |
        | **Household Income (log)** | Log of median household income |
        | **Building Age** | Years since median year built |
        | **Building Height** | Median number of stories |
        """)
