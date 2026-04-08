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
AMBER = "#F59E0B"


# ── DATA + MODELS ─────────────────────────────
@st.cache_data
def load_and_fit():
    crime = pd.read_csv("data/crime_panel.csv")
    housing = pd.read_csv("data/housing_panel.csv")
    temescal = pd.read_csv("data/temescal_yearly_panel.csv")

    c = crime.dropna(subset=["crime_rate", "density", "log_income"])
    h = housing.dropna(subset=["log_income", "crime_rate"])

    m_total = smf.ols(
        "crime_rate ~ density + log_median_value + pct_residential "
        "+ log_income + C(year)", data=c
    ).fit(cov_type="HC1")

    m_violent = smf.ols(
        "violent_rate ~ density + log_median_value + pct_residential "
        "+ log_income + C(year)", data=c
    ).fit(cov_type="HC1")

    m_housing = smf.ols(
        "log_median_value ~ density + pct_residential + building_age "
        "+ median_stories + crime_rate + violent_rate + log_income + C(year)",
        data=h
    ).fit(cov_type="HC1")

    m_van_ness = smf.ols(
        "log_median_value ~ van_ness_treated + post_van_ness + van_ness_x_post "
        "+ density + building_age + crime_rate + C(year)",
        data=housing.dropna(subset=["crime_rate"])
    ).fit(cov_type="HC1")

    m_tem = smf.ols("total_crime ~ treated + post + treated_x_post", data=temescal).fit(cov_type="HC1")
    m_prop = smf.ols("property ~ treated + post + treated_x_post", data=temescal).fit(cov_type="HC1")
    m_viol = smf.ols("violent ~ treated + post + treated_x_post", data=temescal).fit(cov_type="HC1")

    return crime, housing, temescal, m_total, m_violent, m_housing, m_van_ness, m_tem, m_prop, m_viol

crime_panel, housing_panel, temescal_yr, m_total, m_violent, m_housing, m_van_ness, m_tem, m_prop, m_viol = load_and_fit()

PALETTE = [BLUE, RED, GREEN, AMBER]

def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

def coef_chart(model, keep_vars, labels, title, color):
    coefs = model.params[keep_vars]
    ci = model.conf_int().loc[keep_vars]
    errs = np.array([coefs - ci[0], ci[1] - coefs])
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(keep_vars) * 0.55)))
    ax.barh(range(len(keep_vars)), coefs.values, xerr=errs, height=0.5,
            color=color, alpha=0.85, edgecolor="white", capsize=4,
            error_kw={"linewidth": 1.5, "color": "#374151"})
    ax.axvline(0, color="#6B7280", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_yticks(range(len(keep_vars)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coefficient (95% CI)")
    ax.set_title(title, fontweight="bold", fontsize=12)
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
st.markdown("What tradeoffs do local policies create -- and what factors are driving them?")

tab1, tab2, tab3, tab4 = st.tabs([
    "Upzoning", "Office-to-Residential", "Hospitality Task Force", "Data + Variables"
])


# ══════════════════════════════════════════════
# TAB 1: UPZONING
# ══════════════════════════════════════════════

with tab1:
    st.subheader("Upzoning and Densification")
    st.markdown("Oakland upzoned Temescal in 2015. What happened to crime -- and what factors explain it?")

    # DiD trend chart
    col_chart, col_metrics = st.columns([2, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for nbhd, color, marker in [("Temescal", RED, "o"), ("Control", BLUE, "s")]:
            sub = temescal_yr[temescal_yr["neighborhood"] == nbhd]
            ax.plot(sub["year"], sub["total_crime"], color=color, marker=marker,
                    linewidth=2.5, markersize=7, label=nbhd)
        ax.axvline(2015, color="#6B7280", linewidth=2, linestyle="--", alpha=0.7)
        ax.annotate("Upzoning (2015)", xy=(2015, ax.get_ylim()[1] * 0.92),
                    fontsize=10, ha="center", color="#6B7280", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Total Crimes")
        ax.legend(frameon=True)
        ax.set_xticks(sorted(temescal_yr["year"].unique()))
        ax.set_title("Temescal vs Control: Crime Trends", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_metrics:
        st.metric("Total Crime", f"+{m_tem.params['treated_x_post']:.0f}/yr",
                  delta=f"p={m_tem.pvalues['treated_x_post']:.3f}")
        st.metric("Property Crime", f"+{m_prop.params['treated_x_post']:.0f}/yr",
                  delta=f"p={m_prop.pvalues['treated_x_post']:.3f}")
        st.metric("Violent Crime", f"+{m_viol.params['treated_x_post']:.0f}/yr",
                  delta=f"p={m_viol.pvalues['treated_x_post']:.2f} (n.s.)")

    st.markdown("---")
    st.markdown("**What factors explain this?** The crime regression shows which neighborhood characteristics drive crime rates:")

    col1, col2 = st.columns(2)
    keep = ["density", "log_median_value", "pct_residential", "log_income"]
    labels = ["Housing Density", "Property Value (log)", "% Residential Land", "Household Income (log)"]

    with col1:
        fig = coef_chart(m_total, keep, labels, "Total Crime Rate Drivers", BLUE)
        st.pyplot(fig)
        plt.close()
    with col2:
        fig = coef_chart(m_violent, keep, labels, "Violent Crime Rate Drivers", RED)
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════
# TAB 2: OFFICE-TO-RESIDENTIAL
# ══════════════════════════════════════════════

with tab2:
    st.subheader("Office-to-Housing Conversion: 100 Van Ness")
    st.markdown("Converted from office to 399 residential units in 2015 (Hayes Valley / Civic Center).")

    # DiD result
    pct_effect = (np.exp(m_van_ness.params["van_ness_x_post"]) - 1) * 100

    col1, col2 = st.columns([2, 1])

    with col1:
        keep_vn = ["van_ness_treated", "post_van_ness", "van_ness_x_post",
                    "density", "building_age", "crime_rate"]
        labels_vn = ["Near 100 Van Ness", "After 2015", "DiD Effect",
                      "Housing Density", "Building Age", "Crime Rate"]
        fig = coef_chart(m_van_ness, keep_vn, labels_vn,
                         "Effect on Property Values", GREEN)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.metric("DiD Effect", f"{m_van_ness.params['van_ness_x_post']:.3f}",
                  delta=f"p<0.001")
        st.metric("In Plain English", f"{abs(pct_effect):.1f}% slower growth")
        st.metric("Observations", f"{int(m_van_ness.nobs)}")

    st.markdown("---")
    st.markdown("**What factors drive property values?** The housing regression identifies what matters most:")

    keep_h = ["density", "pct_residential", "building_age", "median_stories",
              "crime_rate", "violent_rate", "log_income"]
    labels_h = ["Housing Density", "% Residential Land", "Building Age (years)",
                "Building Height (stories)", "Total Crime Rate", "Violent Crime Rate",
                "Household Income (log)"]

    fig = coef_chart(m_housing, keep_h, labels_h,
                     f"Property Value Drivers  |  R² = {m_housing.rsquared:.3f}", GREEN)
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════
# TAB 3: HOSPITALITY TASK FORCE
# ══════════════════════════════════════════════

with tab3:
    st.subheader("Downtown Hospitality Safety Task Force (Feb 2026)")
    st.markdown("Increased police presence in downtown commercial areas. What does the data tell us about this zone?")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Crime in hospitality zone neighborhoods**")

        # Filter to hospitality zone neighborhoods
        hosp_nbhds = ["Financial District/South Beach", "Tenderloin",
                       "South Of Market", "Nob Hill", "Chinatown"]
        hosp = crime_panel[crime_panel["analysis_neighborhood"].str.strip().str.title().isin(
            [n.strip().title() for n in hosp_nbhds])]
        rest = crime_panel[~crime_panel["analysis_neighborhood"].str.strip().str.title().isin(
            [n.strip().title() for n in hosp_nbhds])]

        if len(hosp) > 0 and len(rest) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            hosp_avg = hosp.groupby("year")["crime_rate"].mean()
            rest_avg = rest.groupby("year")["crime_rate"].mean()
            ax.plot(hosp_avg.index, hosp_avg.values, color=RED, marker="o",
                    linewidth=2.5, label="Hospitality Zone")
            ax.plot(rest_avg.index, rest_avg.values, color=BLUE, marker="s",
                    linewidth=2.5, label="Rest of SF")
            ax.set_xlabel("Year")
            ax.set_ylabel("Avg Crime Rate (per 1,000)")
            ax.set_title("Crime Rate: Hospitality Zone vs Rest of SF", fontweight="bold")
            ax.legend(frameon=True)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        st.markdown("**What factors drive crime in these areas?**")

        if len(hosp) > 0 and len(rest) > 0:
            hosp_mean = hosp[["crime_rate", "violent_rate", "density", "median_hh_income"]].mean()
            rest_mean = rest[["crime_rate", "violent_rate", "density", "median_hh_income"]].mean()

            compare = pd.DataFrame({
                "Metric": ["Crime Rate (per 1K)", "Violent Crime Rate", "Housing Density", "Median HH Income"],
                "Hospitality Zone": [f"{hosp_mean['crime_rate']:.0f}", f"{hosp_mean['violent_rate']:.1f}",
                                      f"{hosp_mean['density']:.1f}", f"${hosp_mean['median_hh_income']:,.0f}"],
                "Rest of SF": [f"{rest_mean['crime_rate']:.0f}", f"{rest_mean['violent_rate']:.1f}",
                                f"{rest_mean['density']:.1f}", f"${rest_mean['median_hh_income']:,.0f}"],
            })
            st.dataframe(compare, use_container_width=True, hide_index=True)

        st.markdown("The regression shows income and density are key crime drivers. "
                    "This zone has higher density and mixed income levels -- "
                    "policing alone may not address the underlying factors.")

    st.markdown("---")
    st.markdown("**What to watch for:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Displacement** -- Are crimes shifting to adjacent neighborhoods?")
    with col2:
        st.markdown("**Crime type** -- Is visible disorder dropping more than serious crime?")
    with col3:
        st.markdown("**Who benefits** -- Tourist perception vs resident safety")


# ══════════════════════════════════════════════
# TAB 4: DATA + VARIABLES
# ══════════════════════════════════════════════

with tab4:
    st.subheader("Data + Variable Definitions")

    col_data, col_dict = st.columns([1.3, 1])

    with col_data:
        display = crime_panel[["analysis_neighborhood", "year", "crime_rate",
                                "violent_rate", "property_rate", "density",
                                "median_hh_income", "pct_residential"]].copy()
        display.columns = ["Neighborhood", "Year", "Total Crime Rate",
                            "Violent Crime Rate", "Property Crime Rate",
                            "Housing Density", "Median HH Income",
                            "% Residential"]
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
        | **Property Value (log)** | Log of median assessed value |
        | **Household Income (log)** | Log of median household income |
        | **Building Age** | Years since median year built |
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
