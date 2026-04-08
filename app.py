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
PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#F59E0B", "#8B5CF6"]


# ── DATA + MODELS ─────────────────────────────
@st.cache_data
def load_and_fit():
    crime = pd.read_csv("data/crime_panel.csv")
    housing = pd.read_csv("data/housing_panel.csv")

    c = crime.dropna(subset=["crime_rate", "density", "log_income"])
    h = housing.dropna(subset=["log_income", "crime_rate"])

    m1a = smf.ols("crime_rate ~ density + log_median_value + pct_residential + log_income + pct_white + pct_hispanic + pct_black + pct_asian + in_hospitality_zone + C(year)", data=c).fit(cov_type="HC1")
    m1c = smf.ols("violent_rate ~ density + log_median_value + pct_residential + log_income + pct_white + pct_hispanic + pct_black + pct_asian + in_hospitality_zone + C(year)", data=c).fit(cov_type="HC1")
    m2a = smf.ols("log_median_value ~ density + pct_residential + building_age + median_stories + crime_rate + violent_rate + log_income + C(year)", data=h).fit(cov_type="HC1")

    return crime, housing, m1a, m1c, m2a

crime_panel, housing_panel, m1a, m1c, m2a = load_and_fit()

def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

def coef_chart(model, keep_vars, labels, title, color):
    coefs = model.params[keep_vars]
    ci = model.conf_int().loc[keep_vars]
    errs = np.array([coefs - ci[0], ci[1] - coefs])
    fig, ax = plt.subplots(figsize=(8, max(3.2, len(keep_vars) * 0.5)))
    ax.barh(range(len(keep_vars)), coefs.values, xerr=errs, height=0.5,
            color=color, alpha=0.85, edgecolor="white", capsize=4,
            error_kw={"linewidth": 1.5, "color": "#374151"})
    ax.axvline(0, color="#6B7280", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_yticks(range(len(keep_vars)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title(title, fontweight="bold")
    for i, var in enumerate(keep_vars):
        pval = model.pvalues[var]
        star = sig_stars(pval)
        if star != "n.s.":
            x_pos = coefs.values[i] + errs[1][i] + abs(coefs.values).max() * 0.03
            ax.text(x_pos, i, star, va="center", fontsize=11, fontweight="bold", color="#DC2626")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# ── LAYOUT ────────────────────────────────────
st.title("Bringing SF Back")
st.markdown("How do local policies impact **housing, public safety, and transit** for SF residents?")
st.markdown("---")

tab_crime, tab_housing, tab_policy, tab_data = st.tabs([
    "Crime Regression", "Housing Regression", "Policy Review", "Data + Variables"
])


# ── TAB 1: CRIME ──────────────────────────────
with tab_crime:
    st.subheader("What Drives Crime Rates?")
    st.markdown(f"OLS | 41 SF neighborhoods | 2018-2023 | R² = {m1a.rsquared:.3f}")

    keep = ["density", "log_median_value", "pct_residential", "log_income",
            "pct_white", "pct_hispanic", "pct_black", "pct_asian", "in_hospitality_zone"]
    labels = ["Housing Density", "Property Value (log)", "% Residential Land",
              "Household Income (log)", "% White", "% Hispanic", "% Black",
              "% Asian", "In Hospitality Zone"]

    col1, col2 = st.columns(2)
    with col1:
        fig = coef_chart(m1a, keep, labels, "Total Crime Rate", PALETTE[0])
        st.pyplot(fig)
        plt.close()
    with col2:
        fig = coef_chart(m1c, keep, labels, "Violent Crime Rate", PALETTE[1])
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Crime R²", f"{m1a.rsquared:.3f}")
    with col2:
        st.metric("Violent Crime R²", f"{m1c.rsquared:.3f}")
    with col3:
        st.metric("Observations", f"{int(m1a.nobs)}")


# ── TAB 2: HOUSING ────────────────────────────
with tab_housing:
    st.subheader("What Drives Property Values?")
    st.markdown(f"OLS | Log median assessed value | 2018-2023 | R² = {m2a.rsquared:.3f}")

    keep_2a = ["density", "pct_residential", "building_age", "median_stories",
                "crime_rate", "violent_rate", "log_income"]
    labels_2a = ["Housing Density", "% Residential Land", "Building Age (years)",
                  "Building Height (stories)", "Total Crime Rate", "Violent Crime Rate",
                  "Household Income (log)"]

    fig = coef_chart(m2a, keep_2a, labels_2a, "Property Value Drivers", PALETTE[2])
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²", f"{m2a.rsquared:.3f}")
    with col2:
        st.metric("Observations", f"{int(m2a.nobs)}")
    with col3:
        st.metric("Strongest Predictor", "Household Income")


# ── TAB 3: POLICY REVIEW ─────────────────────
with tab_policy:
    st.subheader("Policy Tradeoff Summary")
    st.markdown("Each policy creates ripple effects across multiple outcomes.")

    st.markdown("""
    | Policy | Housing Impact | Safety Impact | Transit Impact |
    |--------|--------------|---------------|----------------|
    | **Upzoning / Densification** | Values may rise with density | Property crime increases, violent crime unchanged | More riders on existing routes |
    | **Office-to-Residential** (100 Van Ness) | Surrounding values grow slower | Residents exposed to high-crime commercial area | Increased ridership, equity concern |
    | **Hospitality Task Force** (Feb 2026) | Safety perception affects values | May reduce visible crime, risk displacement | N/A |
    """)

    st.markdown("---")

    st.markdown("**Policy 1: Upzoning**")
    st.markdown("Oakland upzoned Temescal in 2015. Property crime rose significantly; violent crime did not change.")

    st.markdown("**Policy 2: Office-to-Housing Conversion**")
    st.markdown("100 Van Ness (2015) converted office to 399 units. Surrounding property values grew slower than rest of SF.")

    st.markdown("**Policy 3: Hospitality Task Force**")
    st.markdown(f"Downtown hospitality zone already sees +{m1a.params['in_hospitality_zone']:.0f} crimes per 1,000 residents. "
                "Increased policing may reduce visible crime but risks displacement to adjacent neighborhoods.")


# ── TAB 4: DATA + VARIABLES ───────────────────
with tab_data:
    st.subheader("Data + Variable Definitions")

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
        | **Median HH Income** | Median household income ($, Census ACS) |
        | **% Residential** | Share of parcels zoned residential (0-1) |
        | **Hospitality Zone** | 1 = in downtown patrol zone; 0 = outside |
        | **Property Value (log)** | Log of median assessed value |
        | **Household Income (log)** | Log of median household income |
        | **Building Age** | Median year built subtracted from current year |
        | **Building Height** | Median number of stories |
        """)

    st.markdown("---")
    st.markdown("""
    | Dataset | Source | Coverage |
    |---------|--------|----------|
    | Crime (SF) | SFPD Incident Reports | 2018-2023 |
    | Crime (Oakland) | OPD CrimeWatch | 2012-2023 |
    | Housing | SF Assessor | 2018-2023 |
    | Demographics | Census ACS 5-Year | 2013-2023 |
    | Transit | 511 SF Bay API | Real-time |
    """)
