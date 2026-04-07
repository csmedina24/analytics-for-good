"""
Analytics for Good — Interactive Policy Dashboard
SF & Oakland crime, housing, and transit analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title="Analytics for Good",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Style ─────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#F59E0B", "#8B5CF6", "#EC4899"]


# ══════════════════════════════════════════════
# DATA LOADING (cached)
# ══════════════════════════════════════════════

@st.cache_data
def load_data():
    crime = pd.read_csv("data/crime_panel.csv")
    housing = pd.read_csv("data/housing_panel.csv")
    yearly = pd.read_csv("data/temescal_yearly_panel.csv")
    monthly = pd.read_csv("data/temescal_monthly_panel.csv")
    census = pd.read_csv("data/census_income_neighborhoods.csv")
    return crime, housing, yearly, monthly, census

crime_panel, housing_panel, temescal_yr, temescal_mo, census = load_data()


@st.cache_data
def fit_models(crime_df, housing_df, yearly_df):
    """Fit all models once and cache results."""
    c = crime_df.dropna(subset=["crime_rate", "density", "log_income"])

    m1a = smf.ols(
        "crime_rate ~ density + log_median_value + pct_residential "
        "+ log_income + pct_white + pct_hispanic + pct_black + pct_asian "
        "+ in_hospitality_zone + C(year)", data=c
    ).fit(cov_type="HC1")

    m1c = smf.ols(
        "violent_rate ~ density + log_median_value + pct_residential "
        "+ log_income + pct_white + pct_hispanic + pct_black + pct_asian "
        "+ in_hospitality_zone + C(year)", data=c
    ).fit(cov_type="HC1")

    h = housing_df.dropna(subset=["log_income", "crime_rate"])
    m2a = smf.ols(
        "log_median_value ~ density + pct_residential + building_age "
        "+ median_stories + crime_rate + violent_rate + log_income + C(year)",
        data=h
    ).fit(cov_type="HC1")

    m2c = smf.ols(
        "log_median_value ~ van_ness_treated + post_van_ness + van_ness_x_post "
        "+ density + building_age + crime_rate + C(year)",
        data=housing_df.dropna(subset=["crime_rate"])
    ).fit(cov_type="HC1")

    m_tem = smf.ols("total_crime ~ treated + post + treated_x_post", data=yearly_df).fit(cov_type="HC1")
    m_prop = smf.ols("property ~ treated + post + treated_x_post", data=yearly_df).fit(cov_type="HC1")
    m_viol = smf.ols("violent ~ treated + post + treated_x_post", data=yearly_df).fit(cov_type="HC1")
    m_drug = smf.ols("drugs ~ treated + post + treated_x_post", data=yearly_df).fit(cov_type="HC1")

    return m1a, m1c, m2a, m2c, m_tem, m_prop, m_viol, m_drug

m1a, m1c, m2a, m2c, m_tem, m_prop, m_viol, m_drug = fit_models(
    crime_panel, housing_panel, temescal_yr
)


# ══════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════

def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

def coef_chart(model, keep_vars, labels, title, color):
    """Horizontal coefficient plot with 95% CI."""
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


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════

st.sidebar.title("Analytics for Good")
st.sidebar.markdown("**Policy tradeoffs in SF & Oakland**")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Overview",
    "Policy Review",
    "Model 1: Crime Drivers",
    "Model 2: Housing Values",
    "Temescal DiD",
    "Cross-Validation",
    "Policy Simulator",
])

st.sidebar.markdown("---")
st.sidebar.caption("UC Berkeley — Spring 2026")


# ══════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════

if page == "Overview":
    st.title("Bringing SF Back")
    st.subheader("Policy Analysis")

    st.markdown("""
    To what extent are local “Bring SF Back” policies improving quality of life for San Francisco residents, 
    and how do their measurable impacts on housing, transportation, and public safety compared to public narratives 
    about the city’s recovery?

    This dashboard presents three regression models and multiple Difference-in-Differences
    case studies to understand **what really drives** outcomes in Bay Area communities.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model 1: Crime", f"R² = {m1a.rsquared:.2f}",
                   delta="Income is strongest predictor")
    with col2:
        st.metric("Model 2: Housing", f"R² = {m2a.rsquared:.2f}",
                   delta="Density raises values")
    with col3:
        st.metric("Temescal DiD", "+612 crimes/yr",
                   delta="Property crime up, Violent unchanged", delta_color="inverse")

    st.markdown("---")

    # Regression summary
    st.subheader("Regression Models")
    reg_df = pd.DataFrame({
        "Model": ["1A: Total Crime Rate", "1C: Violent Crime Rate",
                   "2A: Property Values"],
        "Method": ["OLS (HC1)", "OLS (HC1)", "OLS (HC1)"],
        "R²": [f"{m1a.rsquared:.3f}", f"{m1c.rsquared:.3f}",
               f"{m2a.rsquared:.3f}"],
        "N": [int(m1a.nobs), int(m1c.nobs), int(m2a.nobs)],
        "Key Finding": [
            f"Income is strongest crime predictor (coef = {m1a.params['log_income']:.1f}***)",
            "Income & racial composition drive violent crime",
            "Income (+) and violent crime (-) drive values",
        ],
    })
    st.dataframe(reg_df, use_container_width=True, hide_index=True)

    # DiD case studies
    st.subheader("Difference-in-Differences Case Studies")
    did_df = pd.DataFrame({
        "Case Study": ["100 Van Ness Conversion", "Temescal Upzoning",
                        "Temescal Upzoning", "Temescal Upzoning"],
        "Outcome": ["Property Values", "Total Crime",
                     "Property Crime", "Violent Crime"],
        "DiD Estimate": [
            f"{m2c.params['van_ness_x_post']:.3f} (log $)",
            f"+{m_tem.params['treated_x_post']:.0f} crimes/yr",
            f"+{m_prop.params['treated_x_post']:.0f} crimes/yr",
            f"+{m_viol.params['treated_x_post']:.0f} crimes/yr"],
        "p-value": [
            f"{m2c.pvalues['van_ness_x_post']:.4f}",
            f"{m_tem.pvalues['treated_x_post']:.4f}",
            f"{m_prop.pvalues['treated_x_post']:.4f}",
            f"{m_viol.pvalues['treated_x_post']:.4f}"],
        "Significant?": [
            "Yes — values grew 13.7% slower",
            "Yes — crime increased post-upzoning",
            "Yes — property crime drove the increase",
            "No — violent crime unchanged"],
    })
    st.dataframe(did_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Data Sources")
    st.markdown("""
    | Dataset | Source | Coverage |
    |---------|--------|----------|
    | Crime (SF) | SFPD Incident Reports | 2018–2023, 1M+ incidents |
    | Crime (Oakland) | OPD CrimeWatch | 2012–2023, 740K incidents |
    | Housing | SF Assessor / Recorder | 2018–2023, 2.8M parcels |
    | Demographics | Census ACS 5-Year | 2013–2023, tract-level |
    | Transit | 511 SF Bay SIRI API | Real-time collection |
    """)


# ══════════════════════════════════════════════
# PAGE: MODEL 1 — CRIME DRIVERS
# ══════════════════════════════════════════════

elif page == "Model 1: Crime Drivers":
    st.title("Model 1: What Drives Crime Rates?")
    st.markdown("""
    Panel regression of **crime rates per 1,000 residents** across 41 SF neighborhoods
    over 6 years (2018–2023). Uses OLS with robust standard errors (HC1) and year
    fixed effects.
    """)

    keep = ["density", "log_median_value", "pct_residential",
            "log_income", "pct_white", "pct_hispanic", "pct_black",
            "pct_asian", "in_hospitality_zone"]
    labels = ["Housing Density", "Property Value (log)", "% Residential Land",
              "Household Income (log)", "% White", "% Hispanic", "% Black",
              "% Asian", "In Hospitality Zone"]

    tab1, tab2, tab3 = st.tabs(["Total Crime", "Violent Crime", "Data"])

    with tab1:
        st.subheader("Model 1A: Total Crime Rate")
        fig = coef_chart(m1a, keep, labels,
                         "What Drives Total Crime Rate?", PALETTE[0])
        st.pyplot(fig)
        plt.close()

        with st.expander("Interpretation"):
            st.markdown(f"""
            - **Household Income (log)** has the strongest effect: a 1% increase in
              neighborhood income is associated with **{m1a.params['log_income']:.1f}
              fewer crimes** per 1,000 residents (p<0.001)
            - **Hospitality Zone** neighborhoods see **+{m1a.params['in_hospitality_zone']:.0f}
              more crimes** per 1,000 (tourism/nightlife effect)
            - **R² = {m1a.rsquared:.3f}** — the model explains {m1a.rsquared*100:.0f}% of
              variation in crime rates
            """)

    with tab2:
        st.subheader("Model 1C: Violent Crime Rate")
        fig = coef_chart(m1c, keep, labels,
                         "What Drives Violent Crime Rate?", PALETTE[1])
        st.pyplot(fig)
        plt.close()

        with st.expander("Interpretation"):
            st.markdown(f"""
            - **R² = {m1c.rsquared:.3f}** — violent crime is even more predictable
            - Income remains the dominant predictor
            - Racial composition variables are significant — reflecting structural
              inequality, not individual behavior
            """)

    with tab3:
        col_data, col_dict = st.columns([1.3, 1])

        with col_data:
            st.subheader("Crime Panel Data")
            display_crime = crime_panel[["analysis_neighborhood", "year", "crime_rate",
                                         "violent_rate", "property_rate", "density",
                                         "median_hh_income", "pct_residential",
                                         "in_hospitality_zone"]].copy()
            display_crime.columns = ["Neighborhood", "Year", "Total Crime Rate",
                                      "Violent Crime Rate", "Property Crime Rate",
                                      "Housing Density", "Median Household Income",
                                      "% Residential Land", "In Hospitality Zone"]
            st.dataframe(display_crime.round(2), use_container_width=True, hide_index=True)

        with col_dict:
            st.subheader("Variable Definitions")
            st.markdown("""
            | Variable | What It Means |
            |----------|--------------|
            | **Neighborhood** | SF Analysis Neighborhood name |
            | **Year** | Calendar year (2018-2023) |
            | **Total Crime Rate** | Total reported crimes per 1,000 residents |
            | **Violent Crime Rate** | Violent crimes (assault, robbery, homicide) per 1,000 residents |
            | **Property Crime Rate** | Property crimes (theft, burglary, vandalism) per 1,000 residents |
            | **Housing Density** | Average housing units per parcel — higher means more multi-unit buildings |
            | **Median Household Income** | Median annual household income from Census ACS ($) |
            | **% Residential Land** | Share of parcels classified as residential (0 to 1) |
            | **In Hospitality Zone** | 1 = neighborhood falls within SF's downtown hospitality patrol zone; 0 = outside |
            """)


# ══════════════════════════════════════════════
# PAGE: MODEL 2 — HOUSING VALUES
# ══════════════════════════════════════════════

elif page == "Model 2: Housing Values":
    st.title("Model 2: What Drives Property Values?")

    tab1, tab2 = st.tabs(["Cross-Sectional Drivers", "100 Van Ness DiD"])

    with tab1:
        st.subheader("Model 2A: Property Value Drivers")
        st.markdown("Log-median assessed value across SF neighborhoods, with year fixed effects.")

        keep_2a = ["density", "pct_residential", "building_age",
                    "median_stories", "crime_rate", "violent_rate", "log_income"]
        labels_2a = ["Housing Density", "% Residential Land", "Building Age (years)",
                      "Building Height (stories)", "Total Crime Rate", "Violent Crime Rate",
                      "Household Income (log)"]

        fig = coef_chart(m2a, keep_2a, labels_2a,
                         "What Drives Property Values?", PALETTE[2])
        st.pyplot(fig)
        plt.close()

        with st.expander("Interpretation"):
            st.markdown(f"""
            - **Income** is the strongest positive predictor — wealthier neighborhoods
              have higher property values
            - **Crime rate** has a negative effect — more crime suppresses values
            - **Density** has a positive effect — denser neighborhoods are more valuable
            - **R² = {m2a.rsquared:.3f}**
            """)

    with tab2:
        st.subheader("Model 2C: 100 Van Ness Office-to-Residential Conversion")
        st.markdown("""
        **Treatment**: Hayes Valley / Civic Center (around 100 Van Ness)
        **Control**: All other SF neighborhoods
        **Intervention**: 2015 conversion of office building to 399 residential units
        """)

        keep_2c = ["van_ness_treated", "post_van_ness", "van_ness_x_post",
                    "density", "building_age", "crime_rate"]
        labels_2c = ["Near 100 Van Ness (treatment)", "After 2015", "DiD Effect (treatment x post)",
                      "Housing Density", "Building Age (years)", "Total Crime Rate"]

        fig = coef_chart(m2c, keep_2c, labels_2c,
                         "100 Van Ness Conversion: Effect on Property Values", PALETTE[4])
        st.pyplot(fig)
        plt.close()

        did_coef = m2c.params["van_ness_x_post"]
        did_p = m2c.pvalues["van_ness_x_post"]
        pct_effect = (np.exp(did_coef) - 1) * 100

        st.error(f"""
        **DiD Estimate: {did_coef:.3f} ({sig_stars(did_p)}, p={did_p:.4f})**

        Property values near 100 Van Ness grew **{abs(pct_effect):.1f}% slower** than
        the rest of SF after the conversion. Adding 399 residential units to a former
        office building moderated the surrounding area's price growth.
        """)


# ══════════════════════════════════════════════
# PAGE: TEMESCAL DiD
# ══════════════════════════════════════════════

elif page == "Temescal DiD":
    st.title("Temescal Upzoning: Difference-in-Differences")
    st.markdown("""
    **Treatment**: Temescal, Oakland (upzoned in 2015)
    **Control**: Laurel / Dimond, Oakland (no upzoning)
    **Parallel trends**: Validated (p=0.56)
    """)

    # Trend lines
    st.subheader("Crime Trends: Temescal vs Control")

    crime_type = st.selectbox("Select crime type",
                               ["total_crime", "property", "violent", "drugs"],
                               format_func=lambda x: x.replace("_", " ").title())

    model_map = {"total_crime": m_tem, "property": m_prop,
                 "violent": m_viol, "drugs": m_drug}
    model = model_map[crime_type]

    fig, ax = plt.subplots(figsize=(10, 5))
    for nbhd, color, marker in [("Temescal", PALETTE[1], "o"), ("Control", PALETTE[0], "s")]:
        sub = temescal_yr[temescal_yr["neighborhood"] == nbhd]
        ax.plot(sub["year"], sub[crime_type], color=color, marker=marker,
                linewidth=2.5, markersize=7, label=nbhd, zorder=3)

    ax.axvline(2015, color="#6B7280", linewidth=2, linestyle="--", alpha=0.7)
    ax.annotate("Upzoning (2015)", xy=(2015, ax.get_ylim()[1] * 0.92),
                fontsize=10, ha="center", color="#6B7280", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Crime Count")
    ax.legend(frameon=True)
    ax.set_xticks(sorted(temescal_yr["year"].unique()))

    did = model.params["treated_x_post"]
    pval = model.pvalues["treated_x_post"]
    ax.set_title(f"DiD Estimate: {did:+,.0f}/year ({sig_stars(pval)}, p={pval:.3f})",
                 fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Four-panel summary
    st.subheader("All Crime Types — DiD Summary")

    col1, col2, col3, col4 = st.columns(4)
    for col, (label, m) in zip([col1, col2, col3, col4],
                                [("Total", m_tem), ("Property", m_prop),
                                 ("Violent", m_viol), ("Drugs", m_drug)]):
        d = m.params["treated_x_post"]
        p = m.pvalues["treated_x_post"]
        with col:
            st.metric(label, f"{d:+,.0f}/yr",
                      delta=f"p={p:.3f} {sig_stars(p)}",
                      delta_color="inverse" if p < 0.05 else "off")

    st.markdown("---")

    with st.expander("Monthly Panel Data"):
        col_data, col_dict = st.columns([1.3, 1])

        with col_data:
            display_tem = temescal_mo.copy()
            display_tem.columns = ["Neighborhood", "Year", "Month", "Is Treatment",
                                    "Post-Upzoning", "Treatment x Post", "Total Crime",
                                    "Violent Crime", "Property Crime", "Drug Crime", "Time Index"]
            st.dataframe(display_tem, use_container_width=True, hide_index=True)

        with col_dict:
            st.markdown("""
            **Variable Definitions**

            | Variable | What It Means |
            |----------|--------------|
            | **Neighborhood** | Temescal (treatment) or Laurel/Dimond (control) |
            | **Is Treatment** | 1 = Temescal (upzoned); 0 = control group |
            | **Post-Upzoning** | 1 = after 2015 upzoning; 0 = before |
            | **Treatment x Post** | The DiD interaction — 1 only for Temescal after 2015 |
            | **Total Crime** | Total reported crimes that month |
            | **Violent Crime** | Assault, robbery, homicide, domestic violence |
            | **Property Crime** | Theft, burglary, vehicle theft, vandalism |
            | **Drug Crime** | Drug-related offenses |
            | **Time Index** | Sequential month number (1 = first month in data) |
            """)


# ══════════════════════════════════════════════
# PAGE: CROSS-VALIDATION
# ══════════════════════════════════════════════

elif page == "Cross-Validation":
    st.title("Two Cities, One Story")
    st.markdown("""
    The SF regression (correlational) and Oakland DiD (causal) independently
    reach the same conclusion: **density drives property crime**.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("SF: Cross-Sectional Evidence")
        keep = ["density", "log_median_value", "log_income", "in_hospitality_zone"]
        labels = ["Housing Density", "Log Property Value",
                  "Log Household Income", "Hospitality Zone"]
        fig = coef_chart(m1a, keep, labels,
                         "SF Regression (41 neighborhoods × 6 years)", PALETTE[0])
        st.pyplot(fig)
        plt.close()
        st.info(f"Density coefficient: **{m1a.params['density']:.2f}** "
                f"(p={m1a.pvalues['density']:.3f})")

    with col2:
        st.subheader("Oakland: Causal Evidence")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        did_data = [
            ("Total Crime", m_tem.params["treated_x_post"],
             m_tem.bse["treated_x_post"] * 1.96, m_tem.pvalues["treated_x_post"]),
            ("Property Crime", m_prop.params["treated_x_post"],
             m_prop.bse["treated_x_post"] * 1.96, m_prop.pvalues["treated_x_post"]),
            ("Violent Crime", m_viol.params["treated_x_post"],
             m_viol.bse["treated_x_post"] * 1.96, m_viol.pvalues["treated_x_post"]),
        ]
        labels_did = [d[0] for d in did_data]
        vals = [d[1] for d in did_data]
        errs = [d[2] for d in did_data]
        colors = [PALETTE[1] if d[3] < 0.05 else "#D1D5DB" for d in did_data]

        ax.barh(range(len(labels_did)), vals, xerr=errs, height=0.5,
                color=colors, alpha=0.85, edgecolor="white", capsize=4,
                error_kw={"linewidth": 1.5, "color": "#374151"})
        ax.axvline(0, color="#6B7280", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_yticks(range(len(labels_did)))
        ax.set_yticklabels(labels_did)
        ax.set_xlabel("DiD (additional crimes/year)")
        ax.set_title("Temescal Upzoning DiD (2012–2023)", fontweight="bold")
        ax.invert_yaxis()
        for i, d in enumerate(did_data):
            ax.text(d[1] + d[2] + 30, i, sig_stars(d[3]),
                    va="center", fontsize=11, fontweight="bold",
                    color="#DC2626" if d[3] < 0.05 else "#9CA3AF")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info(f"Property crime DiD: **+{m_prop.params['treated_x_post']:.0f}/year** "
                f"(p={m_prop.pvalues['treated_x_post']:.3f})")

    st.markdown("---")

    # Policy tradeoff summary
    st.subheader("The Tradeoff")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**Density --> Higher Property Values**\n\n"
                    f"Housing regression coef: {m2a.params['density']:+.4f}***")
    with col2:
        st.error("**Density --> Higher Property Crime**\n\n"
                 f"Temescal DiD: +{m_prop.params['treated_x_post']:.0f} crimes/yr**")
    with col3:
        st.warning("**Density --> No Change in Violent Crime**\n\n"
                   f"Temescal DiD: +{m_viol.params['treated_x_post']:.0f} (p={m_viol.pvalues['treated_x_post']:.2f}, n.s.)")


# ══════════════════════════════════════════════
# PAGE: POLICY SIMULATOR
# ══════════════════════════════════════════════

elif page == "Policy Simulator":
    st.title("Policy Tradeoff Simulator")
    st.markdown("""
    Adjust neighborhood characteristics and see predicted impacts on **crime**
    and **housing values**. The predictions come directly from the regression
    coefficients — showing that **there is no free lunch in policy**.
    """)

    st.markdown("---")

    # Sliders
    col_input, col_spacer, col_output = st.columns([1, 0.1, 1.2])

    with col_input:
        st.subheader("Adjust Inputs")

        # Get baseline values (medians from data)
        c = crime_panel.dropna(subset=["density", "log_income"])

        density = st.slider(
            "Housing Density",
            min_value=0.5, max_value=30.0,
            value=float(c["density"].median()), step=0.5,
            help="Average housing units per parcel. Higher = more apartments/condos, lower = single-family homes.")

        income = st.slider(
            "Median Household Income",
            min_value=30000, max_value=250000,
            value=int(np.exp(c["log_income"].median())), step=5000,
            format="$%d",
            help="Median annual household income in the neighborhood (from Census ACS).")

        pct_res = st.slider(
            "Share of Residential Land",
            min_value=0.0, max_value=1.0,
            value=float(c["pct_residential"].median()), step=0.05,
            format="%.0f%%",
            help="Proportion of parcels used for housing (vs commercial/industrial). 1.0 = fully residential.")

        hosp_zone = st.toggle("Located in Hospitality Zone", value=False,
            help="SF's downtown tourism/nightlife patrol zone (FiDi, Tenderloin, Union Square, SOMA).")

        prop_value = st.slider(
            "Median Property Value",
            min_value=200000, max_value=3000000,
            value=int(np.exp(c["log_median_value"].median())), step=50000,
            format="$%d",
            help="Median assessed property value in the neighborhood from SF Assessor data.")

        st.caption("*All sliders start at the SF neighborhood median*")

    with col_output:
        st.subheader("Predicted Outcomes")

        # Calculate predictions using Model 1A coefficients
        log_inc = np.log(income)
        log_val = np.log(prop_value)

        # Crime prediction (simplified — using key coefficients only)
        pred_crime = (m1a.params.get("Intercept", 0)
                      + m1a.params["density"] * density
                      + m1a.params["log_median_value"] * log_val
                      + m1a.params["pct_residential"] * pct_res
                      + m1a.params["log_income"] * log_inc
                      + m1a.params["in_hospitality_zone"] * int(hosp_zone))

        # Violent crime prediction
        pred_violent = (m1c.params.get("Intercept", 0)
                        + m1c.params["density"] * density
                        + m1c.params["log_median_value"] * log_val
                        + m1c.params["pct_residential"] * pct_res
                        + m1c.params["log_income"] * log_inc
                        + m1c.params["in_hospitality_zone"] * int(hosp_zone))

        # Housing prediction
        pred_log_value = (m2a.params.get("Intercept", 0)
                          + m2a.params["density"] * density
                          + m2a.params["pct_residential"] * pct_res
                          + m2a.params["crime_rate"] * max(pred_crime, 0)
                          + m2a.params["log_income"] * log_inc)

        pred_value = np.exp(pred_log_value)

        # Baseline predictions (at median values)
        base_crime = float(c["crime_rate"].median())
        base_violent = float(c["violent_rate"].median())
        base_value = float(np.exp(
            housing_panel.dropna(subset=["log_median_value"])["log_median_value"].median()
        ))

        # Display metrics
        crime_delta = pred_crime - base_crime
        violent_delta = pred_violent - base_violent
        value_delta = ((pred_value - base_value) / base_value) * 100

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Crime Rate",
                      f"{max(pred_crime, 0):.0f} per 1K",
                      delta=f"{crime_delta:+.0f} vs median",
                      delta_color="inverse")
        with m2:
            st.metric("Violent Crime Rate",
                      f"{max(pred_violent, 0):.0f} per 1K",
                      delta=f"{violent_delta:+.0f} vs median",
                      delta_color="inverse")
        with m3:
            st.metric("Predicted Property Value",
                      f"${pred_value:,.0f}",
                      delta=f"{value_delta:+.1f}% vs median")

        st.markdown("---")

        # Visual bar comparison
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

        for ax, (label, pred, base, color, invert) in zip(axes, [
            ("Crime Rate", max(pred_crime, 0), base_crime, PALETTE[1], True),
            ("Violent Rate", max(pred_violent, 0), base_violent, PALETTE[4], True),
            ("Property Value", pred_value / 1000, base_value / 1000, PALETTE[2], False),
        ]):
            bars = ax.bar(["SF Median", "Your\nScenario"], [base, pred],
                          color=["#D1D5DB", color], alpha=0.85, edgecolor="white", width=0.5)
            ax.set_title(label, fontweight="bold", fontsize=11)
            ax.set_ylabel("per 1K res." if "Rate" in label else "$K")
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Tradeoff callout
    st.markdown("---")
    if density > c["density"].median() * 1.3:
        st.warning("**Tradeoff alert**: High density is boosting predicted property "
                   "values but also increasing predicted crime. This is the core tension "
                   "policymakers face with upzoning.")
    if hosp_zone:
        st.warning("**Hospitality zone** adds significant crime — tourism and nightlife "
                   "bring economic activity but also public safety costs.")
    if income > 150000:
        st.success("High income is the strongest crime reducer — but income can't simply "
                   "be legislated. It reflects decades of investment and structural advantage.")


# ══════════════════════════════════════════════
# PAGE: POLICY REVIEW
# ══════════════════════════════════════════════

elif page == "Policy Review":
    st.title("Policy Review")
    st.markdown("""
    Each "Bring SF Back" policy targets a specific urban challenge but creates ripple effects
    across housing, public safety, and transit. Below we link each policy to the specific
    analyses that measure its impacts — and identify where those impacts create tradeoffs.
    """)

    st.markdown("---")

    # ── POLICY 1: FAMILY ZONING / UPZONING ──
    st.subheader("Policy 1: Upzoning and Densification")

    st.markdown("""
    **What it is**: At the end of 2014, Oakland adopted a housing element (2015-2023) focusing
    on upzoning and densification — enabling taller buildings and denser communities in areas
    previously zoned for single-family housing. Temescal, once an area of primarily single-family
    bungalows, transitioned into a popular, dense urban-suburban area with a high percentage of renters.

    **Why Oakland**: Similar urban environment, inequality patterns, and crime dynamics to SF —
    serves as a model for what SF upzoning policies could produce.
    """)

    st.markdown("**Linked Analyses:**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Public Safety Impact**")
        st.markdown("""
        - **Analysis**: Temescal DiD (Oakland crime data, 2012-2023)
        - **Treatment**: Temescal (upzoned) vs Laurel/Dimond (control)
        - **Finding**: +602 property crimes/year (p=0.002), no change in violent crime (p=0.89)
        - **Interpretation**: Upzoning attracted opportunity crime (theft, burglary, vehicle break-ins)
          tied to increased density and foot traffic — not interpersonal violence
        """)

    with col2:
        st.markdown("**Housing Value Impact**")
        st.markdown("""
        - **Analysis**: SOMA vs Potrero Hill DiD (Model 2B)
        - **Treatment**: SOMA (densified with SOMA Grand 2009, NEMA 2014) vs Potrero Hill
          (primarily single-family, no densification shift)
        - **Finding**: SOMA vs Potrero Hill coefficient was not statistically significant (p=0.146)
        - **Interpretation**: Large-scale residential development in SOMA did not produce a
          significantly different housing value trajectory compared to Potrero Hill
        """)

    with st.expander("Regression evidence supporting this policy review"):
        st.markdown(f"""
        - **Model 1A** (SF crime regression): density coefficient = +{m1a.params['density']:.2f}
          (p={m1a.pvalues['density']:.3f}) — density is associated with higher crime rates,
          consistent with the Temescal causal finding
        - **Model 2A** (SF housing regression): density coefficient = +{m2a.params['density']:.4f}
          (p={m2a.pvalues['density']:.3f}) — density effect on values is positive but not
          individually significant with controls
        - **Temescal parallel trends**: validated (p=0.56), confirming the DiD is credible
        """)

    st.markdown("---")

    # ── POLICY 2: OFFICE TO HOUSING CONVERSION ──
    st.subheader("Policy 2: Downtown Office-to-Housing Conversion")

    st.markdown("""
    **What it is**: Conversion of office space to residential housing has been rare in SF,
    but 100 Van Ness — completed in 2015 — is the prime example. The former office tower at
    the intersection of Hayes Valley and Civic Center was converted into 399 residential units.

    **Why it matters**: As SF's downtown faces high office vacancy post-pandemic, this policy
    is increasingly relevant. 100 Van Ness provides a case study of what happens when you add
    hundreds of residents to a commercial corridor.
    """)

    st.markdown("**Linked Analyses:**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Housing Value Impact**")
        st.markdown(f"""
        - **Analysis**: 100 Van Ness DiD (Model 2C)
        - **Treatment**: Hayes Valley / Civic Center vs rest of SF
        - **Finding**: DiD = {m2c.params['van_ness_x_post']:.3f} (p<0.001***)
        - **Interpretation**: Property values near 100 Van Ness grew 13.7% slower than the rest
          of SF — the conversion moderated surrounding price growth
        """)

    with col2:
        st.markdown("**Transit Impact**")
        st.markdown("""
        - **Analysis**: Model 3 — Transit Reliability (data collection in progress)
        - **Route**: 49-Van Ness, starting at Van Ness & Market
        - **Question**: How did adding ~400 residents to the Van Ness corridor impact Muni
          headways? Does location north vs south of Market affect who bears the burden?
        - **Equity angle**: Neighborhoods south of Market are generally lower income and
          working class; north of Market income increases (Russian Hill, Cow Hollow)
        - **Status**: Actively collecting real-time headway data via 511 API
        """)

    with col3:
        st.markdown("**Public Safety Impact**")
        st.markdown(f"""
        - **Analysis**: Model 1A — Crime regression
        - **Hospitality zone indicator** captures the downtown commercial area
          around 100 Van Ness
        - **Finding**: Hospitality zone = +{m1a.params['in_hospitality_zone']:.0f} crimes
          per 1,000 residents (p<0.001***)
        - **Context**: Adding residents to high-crime commercial corridors exposes
          them to elevated crime rates
        """)

    with st.expander("Regression evidence supporting this policy review"):
        st.markdown(f"""
        - **Model 2C**: van_ness_x_post = {m2c.params['van_ness_x_post']:.4f} (p<0.001) —
          statistically significant slowdown in value growth
        - **Model 2A**: violent_rate coefficient = {m2a.params['violent_rate']:.4f} (p<0.001) —
          violent crime suppresses property values, relevant to the high-crime downtown corridor
        - **Caveat**: Treatment area also affected by COVID downtown impacts and homelessness crisis
        """)

    st.markdown("---")

    # ── POLICY 3: HOSPITALITY SAFETY TASK FORCE ──
    st.subheader("Policy 3: Downtown Hospitality Safety Task Force (Feb 2026)")

    st.markdown("""
    **What it is**: Launched in early February 2026, this task force aims to improve safety
    in downtown SF through increased police presence in high-volume commercial areas. The
    patrol zone covers the "hospitality zone" — from Bush St to South of Market, encompassing
    Union Square, Yerba Buena, Moscone Center, and the Financial District.

    **The equity question**: While increased police presence may improve visible indicators of
    safety and potentially enhance visitor experience, its effects on residents are uneven.
    Pre-existing disparities in perceived safety, combined with the potential for crime
    displacement, suggest that the policy may not equitably improve safety outcomes across
    communities.
    """)

    st.markdown("**Linked Analyses:**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**What the regression tells us about this zone**")
        st.markdown(f"""
        - **Analysis**: Model 1A — the `in_hospitality_zone` indicator
        - **Finding**: Neighborhoods in the hospitality zone already experience
          +{m1a.params['in_hospitality_zone']:.0f} total crimes and
          +{m1c.params['in_hospitality_zone']:.1f} violent crimes per 1,000 residents
          compared to the rest of SF (both p<0.001)
        - **Income effect**: Log income coefficient = {m1a.params['log_income']:.1f} —
          the strongest crime predictor. The hospitality zone contains both high-income
          (Financial District) and low-income (Tenderloin) neighborhoods, creating
          stark disparities within the zone itself
        """)

    with col2:
        st.markdown("**What to watch for (early indicators)**")
        st.markdown("""
        - **Crime frequency**: Compare Jan 2026 (pre-task force) to Feb-Apr 2026 (post)
          within the hospitality zone vs surrounding neighborhoods
        - **Crime type**: Is the task force reducing visible disorder crimes (drugs,
          vandalism) or more serious offenses (assault, robbery)?
        - **Time of day**: When are crimes occurring — are daytime tourist-hours
          improving while nighttime stays the same?
        - **Displacement**: Are crimes shifting to adjacent neighborhoods (SOMA,
          Tenderloin edges, Civic Center) rather than actually decreasing?
        - **Who benefits**: Tourist perceptions may improve (Super Bowl feedback)
          while resident safety in surrounding lower-income neighborhoods may not
        """)

    with st.expander("Demographic context of the hospitality zone"):
        st.markdown("""
        The hospitality zone and surrounding neighborhoods include:

        | Neighborhood | Character | Key demographic |
        |---|---|---|
        | Financial District / South Beach | High-income commercial | Young professionals, high median income |
        | Tenderloin | Low-income residential | High poverty rate, large homeless population, diverse |
        | South of Market (SOMA) | Mixed commercial/residential | Gentrifying, income disparity within neighborhood |
        | Nob Hill | Upper-income residential | Older, wealthier, predominantly white |
        | Chinatown | Dense residential/commercial | Lower income, predominantly Asian, elderly population |

        The task force patrols the same zone — but the people living in and around it have
        vastly different relationships to safety, policing, and public space. A policy that
        makes Union Square feel safer for tourists may have very different implications for
        Tenderloin residents.
        """)

    st.markdown("---")

    # ── SYNTHESIS ──
    st.subheader("Synthesis: The Tradeoff Framework")
    st.markdown("""
    Every policy we analyzed impacts multiple outcomes simultaneously:

    | Policy | Housing | Public Safety | Transit | Equity |
    |--------|---------|---------------|---------|--------|
    | **Upzoning** (Temescal) | Values may rise with density | Property crime increases, violent crime unchanged | More riders on existing routes | New renters vs existing homeowners |
    | **Office-to-Residential** (100 Van Ness) | Surrounding values grow slower | Residents exposed to high-crime commercial area | Increased ridership on Van Ness corridor | North vs south of Market disparity |
    | **Hospitality Task Force** | Indirect — safety perception affects values | May reduce visible crime but risk displacement | N/A | Tourist safety vs resident safety; income/racial disparities in policing |

    **The core finding**: There is no policy that improves all outcomes for all people.
    Upzoning brings housing but attracts crime. Office conversions moderate prices but
    strain transit. Policing improves visitor perception but may not equitably serve
    residents. Effective policymaking requires acknowledging these tradeoffs explicitly.
    """)


# ══════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Data**: SFPD, OPD, SF Assessor,\n Census ACS, 511 SF Bay

**Methods**: OLS (HC1), DiD
""")
