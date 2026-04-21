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
import json
import urllib.request
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Bringing SF Back", layout="wide")
sns.set_theme(style="whitegrid", font_scale=1.05)
BLUE = "#2563EB"
RED = "#DC2626"
GREEN = "#16A34A"
AMBER = "#F59E0B"
PURPLE = "#7C3AED"



# ── DATA + MODELS ─────────────────────────────
@st.cache_data
def load_and_fit():
    crime = pd.read_csv("data/crime_panel.csv")
    housing = pd.read_csv("data/housing_panel.csv")
    temescal = pd.read_csv("data/temescal_yearly_panel.csv")
    temescal_filtered = temescal[temescal["year"] <= 2020].copy()

    survey_race = pd.read_csv("data/survey_safety_by_race.csv")
    survey_trend = pd.read_csv("data/survey_safety_trend.csv")
    hosp_demo = pd.read_csv("data/hospitality_zone_demographics.csv")
    hosp_monthly = pd.read_csv("data/hospitality_monthly_crime.csv")

    muni_trend = pd.read_csv("data/survey_muni_trend.csv")
    muni_district = pd.read_csv("data/survey_muni_by_district.csv")
    muni_race = pd.read_csv("data/survey_muni_by_race.csv")

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

    m_tem = smf.ols("total_crime ~ treated + post + treated_x_post",
                     data=temescal_filtered).fit(cov_type="HC1")
    m_prop = smf.ols("property ~ treated + post + treated_x_post",
                      data=temescal_filtered).fit(cov_type="HC1")
    m_viol = smf.ols("violent ~ treated + post + treated_x_post",
                      data=temescal_filtered).fit(cov_type="HC1")

    # Hospitality Task Force DiD
    disp = pd.read_csv("data/displacement_crime.csv")
    disp["treated"] = (disp["zone"] == "Hospitality Zone").astype(int)
    disp["post"] = (disp["year_month"] >= "2025-02").astype(int)
    disp["treated_x_post"] = disp["treated"] * disp["post"]
    disp["t"] = disp.groupby("zone").cumcount()

    m_hosp_total = smf.ols(
        "total_crimes ~ treated + post + treated_x_post", data=disp
    ).fit(cov_type="HC1")
    m_hosp_drug = smf.ols(
        "drug_offenses ~ treated + post + treated_x_post", data=disp
    ).fit(cov_type="HC1")
    m_hosp_dispatch = smf.ols(
        "dispatch_calls ~ treated + post + treated_x_post", data=disp
    ).fit(cov_type="HC1")
    m_hosp_trend = smf.ols(
        "total_crimes ~ treated + post + treated_x_post + t", data=disp
    ).fit(cov_type="HC1")

    return (crime, housing, temescal, temescal_filtered,
            m_total, m_violent, m_housing, m_van_ness, m_tem, m_prop, m_viol,
            survey_race, survey_trend, hosp_demo, hosp_monthly,
            muni_trend, muni_district, muni_race,
            m_hosp_total, m_hosp_drug, m_hosp_dispatch, m_hosp_trend)

(crime_panel, housing_panel, temescal_yr, temescal_filt,
 m_total, m_violent, m_housing, m_van_ness, m_tem, m_prop, m_viol,
 survey_race, survey_trend, hosp_demo, hosp_monthly,
 muni_trend, muni_district, muni_race,
 m_hosp_total, m_hosp_drug, m_hosp_dispatch, m_hosp_trend) = load_and_fit()




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

tab1, tab2, tab3 = st.tabs([
    "Office-to-Residential & DRD",
    "Hospitality Task Force", "Data + Variables"
])


# ══════════════════════════════════════════════
# TAB 1: OFFICE-TO-RESIDENTIAL & DRD
# ══════════════════════════════════════════════

with tab1:
    st.subheader("Downtown Revitalization District (DRD)")
    st.markdown("AB 2488 (2024) authorized the Downtown Revitalization District, amended by "
                "AB 1445 (2025), with the Financing Plan adopted February 12, 2026. "
                "This analysis examines the policy's structure, its practical challenges "
                "and benefits, and what may be driving it beyond the stated goal of "
                "'increasing housing and decreasing vacancy.'")

    # ── Legislative History ────────────────────
    st.markdown("#### How We Got Here")
    st.markdown(
        "| Year | Action | Key Players |\n"
        "|------|--------|-------------|\n"
        "| **2024** | **AB 2488** — establishes the DRD framework in state law | "
        "Mayor London Breed + Assemblymember Matt Haney |\n"
        "| **Mar 2025** | **Ordinance 20-25** — waives SF's inclusionary housing (Sec. 415) "
        "and impact fees for C-3 conversions (first 7M sq ft) | Mayor Daniel Lurie |\n"
        "| **2025** | **AB 1445** — amends AB 2488 | Assemblymember Matt Haney |\n"
        "| **Feb 2026** | **Financing Plan adopted** — defines boundaries, tax increment "
        "allocation, and $1.22B district cap | Mayor Daniel Lurie |"
    )
    st.caption("The DRD policy framework was created under the Breed administration. "
               "The Lurie administration inherited it and made key implementation decisions — "
               "including waiving local affordable housing requirements and adopting the financing plan.")

    st.markdown("---")

    # ── Stated Goal: Mayor Lurie ─────────────
    st.markdown("#### The Stated Goal")
    st.info(
        "\"Through our Heart of the City plan, our administration is accelerating downtown's "
        "recovery by activating our public spaces, prioritizing safe and clean streets, and "
        "creating a downtown where people live, work, play, and learn. This new financing "
        "district will support office-to-residential conversions and help turn empty office "
        "buildings into thousands of new homes—helping us add more housing in San Francisco "
        "and delivering on a promise in our Heart of the City plan.\"\n\n"
        "— **Mayor Daniel Lurie**, DRD Press Release (Feb 12, 2026)"
    )
    st.caption(
        "The Heart of the City directive (Sept 2025) frames downtown's transformation as "
        "driving citywide revenue — arguing that \"the ideas, innovation, and revenue generated "
        "downtown fund the services that keep the whole city running.\" "
        "([Source: Office of the Mayor, Sept 2025](https://www.sf.gov/news/mayor-lurie-announces-heart-city-plan))"
    )

    # ── DRD Boundary Map ─────────────────────
    st.markdown("#### District Boundaries")
    col_map_l, col_map, col_map_r = st.columns([1, 2, 1])
    with col_map:
        st.image("data/drd_boundary_map.png", use_container_width=True)
    st.caption("Approximate boundary based on AB 2488 §62450(h) and the Financing Plan (Feb 2026). "
               "Covers downtown C-3 zoning districts.")

    st.markdown("---")

    # ── Load pipeline data ────────────────────
    pipeline = pd.read_csv("data/large_development_projects.csv")
    DRD_NBHDS = ["Financial District/South Beach", "Tenderloin", "South of Market",
                  "Chinatown", "Nob Hill"]
    dt_pipe = pipeline[pipeline["neighborhoods_analysis_boundaries"].isin(DRD_NBHDS)].copy()

    off2res = dt_pipe[
        (dt_pipe["existing_use"].str.contains("office", case=False, na=False)) &
        (dt_pipe["proposed_use"].str.contains("apart|resid|dwell|condo", case=False, na=False))
    ].copy()
    off2res["address"] = off2res["street_number"].astype(str) + " " + off2res["street_name"]
    status_rank = {"complete": 0, "issued": 1, "filed": 2, "expired": 3,
                   "cancelled": 4, "withdrawn": 5}
    off2res["status_rank"] = off2res["status"].map(status_rank).fillna(6)
    off2res_dedup = off2res.sort_values("status_rank").drop_duplicates("address", keep="first")
    completed_conv = off2res_dedup[off2res_dedup["status"] == "complete"]
    active_conv = off2res_dedup[off2res_dedup["status"].isin(["filed", "issued"])]

    # ── Load price data ──────────────────────
    price_df = pd.read_csv("data/sf_commercial_residential_prices.csv")
    price_df["effective_office_rent"] = (price_df["office_rent_psf"]
                                         * (1 - price_df["office_vacancy_pct"] / 100))

    # ══════════════════════════════════════════════════════
    # PART 1: WHAT IS THE DRD?
    # ══════════════════════════════════════════════════════

    st.markdown("## Part 1: What Is the DRD?")
    st.markdown("Understanding the policy, its structure, and the problem it claims to solve.")

    st.markdown("---")

    st.markdown("### Program Overview")

    st.markdown("""
    The DRD finances commercial-to-residential conversions by returning a share of the
    property tax increment generated by the project for up to **30 years**.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Key eligibility criteria:**
        - Located within the District boundaries (downtown C-3 zoning districts)
        - Convert existing Commercial space to Residential use
        - At least **60% of gross floor area** must be Residential
        - Must enroll **before** building permit issuance
        - Enrollment deadline: **December 31, 2032**
        """)
    with col2:
        st.markdown("**How the money flows:**")
        st.markdown(
            "When a developer converts an office building to housing, the property's "
            "value goes up — and so do its property taxes. Normally that new tax revenue "
            "goes to the city. Under the DRD, the city **returns up to 64.59%** of that "
            "increase back to the developer each year for up to **30 years** to help "
            "offset conversion costs."
        )

    # ── Financial Flow Diagram ──────────────
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.set_axis_off()

    # Boxes
    boxes = [
        (0.3, 1.2, "Office → Housing\nConversion", "#E0E7FF", BLUE),
        (3.3, 1.2, "Property Value\nIncreases", "#FEF3C7", AMBER),
        (6.3, 1.2, "New Tax Revenue\nGenerated", "#D1FAE5", GREEN),
    ]
    for x, y, text, fc, ec in boxes:
        rect = plt.Rectangle((x, y), 2.4, 1.4, facecolor=fc, edgecolor=ec,
                              linewidth=2, zorder=2, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + 1.2, y + 0.7, text, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#1F2937", zorder=3)

    # Arrows between boxes
    arrow_style = dict(arrowstyle="->, head_width=0.3, head_length=0.15",
                       color="#6B7280", lw=2)
    ax.annotate("", xy=(3.2, 1.9), xytext=(2.8, 1.9), arrowprops=arrow_style)
    ax.annotate("", xy=(6.2, 1.9), xytext=(5.8, 1.9), arrowprops=arrow_style)

    # Split arrow from tax revenue
    ax.annotate("", xy=(9.5, 2.6), xytext=(8.8, 2.2), arrowprops=arrow_style)
    ax.annotate("", xy=(9.5, 1.0), xytext=(8.8, 1.4), arrowprops=arrow_style)

    # Labels for the split
    ax.text(9.6, 2.65, "64.59% → Developer", fontsize=9, fontweight="bold",
            color=BLUE, va="center")
    ax.text(9.6, 2.3, "(up to 30 years)", fontsize=8, color="#6B7280", va="center")
    ax.text(9.6, 1.0, "35.41% → City", fontsize=9, fontweight="bold",
            color=GREEN, va="center")
    ax.text(9.6, 0.65, "(schools, transit, services)", fontsize=8,
            color="#6B7280", va="center")

    ax.set_title("How the DRD Financial Structure Works", fontweight="bold",
                 fontsize=12, pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        f"**District-wide cap:** $1.22 billion in total allocated tax revenue &nbsp;|&nbsp; "
        f"**Admin costs:** capped at 5% of tax revenues"
    )

    st.markdown("---")

    # ── Affordability Tiers ──────────────────
    st.markdown("### Affordable Housing Requirements: Three Tiers")
    st.markdown("The DRD creates a phased affordability structure. Projects that enroll "
                "early face fewer requirements -- creating a first-come-first-served race.")

    # Stacked bar showing the three tiers
    fig, ax = plt.subplots(figsize=(10, 2.5))
    tier1_end = 1.5
    tier2_end = 7.0
    tier3_end = 10.0

    ax.barh(0, tier1_end, left=0, height=0.5, color=GREEN, alpha=0.85,
            edgecolor="white", linewidth=2)
    ax.barh(0, tier2_end - tier1_end, left=tier1_end, height=0.5, color=AMBER,
            alpha=0.85, edgecolor="white", linewidth=2)
    ax.barh(0, tier3_end - tier2_end, left=tier2_end, height=0.5, color=RED,
            alpha=0.85, edgecolor="white", linewidth=2)

    ax.text(tier1_end / 2, 0, "Tier 1\nNo additional\naffordability",
            ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(tier1_end + (tier2_end - tier1_end) / 2, 0,
            "Tier 2\nState minimums only\n(5-10% affordable)",
            ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(tier2_end + (tier3_end - tier2_end) / 2, 0,
            "Tier 3\nState + local\ninclusionary",
            ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    ax.axvline(tier1_end, color="#374151", linewidth=2, linestyle="-")
    ax.axvline(tier2_end, color="#374151", linewidth=2, linestyle="-")
    ax.text(tier1_end, -0.45, "1.5M sq ft\n(~1,875 units)", ha="center",
            fontsize=9, fontweight="bold", color="#374151")
    ax.text(tier2_end, -0.45, "7M sq ft\n(~8,750 units)", ha="center",
            fontsize=9, fontweight="bold", color="#374151")

    ax.set_xlim(0, tier3_end)
    ax.set_ylim(-0.8, 0.6)
    ax.set_xlabel("Cumulative Enrolled Conversion Square Footage (millions)", fontsize=10)
    ax.set_yticks([])
    ax.set_title("DRD Affordability Thresholds: Who Has to Build Affordable Units?",
                 fontweight="bold", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Tier 1: 0 - 1.5M sq ft**")
        st.markdown("No state affordability requirements (AB 2488). Local inclusionary "
                    "(Sec. 415) waived in C-3 districts by Ordinance 20-25 (Mar 2025). "
                    "These projects can be **100% market-rate**.")
    with col2:
        st.markdown("**Tier 2: 1.5M - 7M sq ft**")
        st.markdown("State minimums kick in -- one of:\n"
                    "- 5% of rental units at **50% AMI** (55 yrs)\n"
                    "- 10% of rental units at **80% AMI** (55 yrs)\n"
                    "- 10% of for-sale units at **120% AMI** (45 yrs)\n\n"
                    "Local inclusionary (Sec. 415) still waived under Ordinance 20-25.")
    with col3:
        st.markdown("**Tier 3: 7M+ sq ft**")
        st.markdown("Both state minimums **and** SF's local inclusionary housing "
                    "requirements (Sec. 415) apply. The highest affordability bar.")

    st.caption("Tracking is first-come-first-served based on Notice of Eligibility date. "
               "Projects that miss the 18-month deadline for a Planning Approval Letter "
               "get dropped to the bottom of the list and may lose their tier exemption.")

    st.markdown("---")

    # ── The Starting Point: Zero Conversions ─
    st.markdown("### The Starting Point: Zero Completed Conversions")

    st.error("**As of February 2026, San Francisco has zero completed office-to-residential "
             "conversions downtown.** Despite years of discussion, high construction costs, "
             "building code requirements, and zoning barriers have prevented any projects from "
             "finishing. The DRD program is designed to break this deadlock by providing financial "
             "incentives. "
             "([SF Chronicle, Feb 2026](https://www.sfchronicle.com/sf/article/downtown-office-conversions-21345575.php))")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Completed Conversions", "0",
                  delta="none as of Feb 2026")
    with col2:
        st.metric("In Pipeline (filed/issued)", f"{len(active_conv)}",
                  delta=f"{active_conv['proposed_units'].sum():.0f} proposed units")
    with col3:
        est_units_at_threshold = 1_500_000 / 800
        st.metric("1.5M Sq Ft Threshold",
                  f"~{est_units_at_threshold:,.0f} units",
                  delta="zero affordability add-on")
    with col4:
        st.metric("Max Tax Increment", "$1.22B",
                  delta="over district lifetime")

    # ══════════════════════════════════════════════════════
    # PART 2: THE REALITY CHECK
    # ══════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown("## Part 2: The DRD in Practice — Transit Analysis")
    st.markdown("New downtown residents will depend on Muni. "
                "How much capacity does the system have to absorb them?")

    st.markdown("---")

    # ── Transit Impact Model ─────────────────
    st.markdown("### Transit Impact: Modeling the Load")

    with st.expander("Methodology: How We Derived Transit Impact Predictions", expanded=False):
        st.markdown("""
        **Objective:** Estimate how new residential units from DRD conversions would affect
        peak-hour crowding on Muni routes serving downtown.

        **Data sources:**
        - **Ridership:** SFMTA monthly ridership reports (2019-present) for daily boarding counts
          by route.
        - **Service frequency & capacity:** SFMTA GTFS schedule data and 511 SF Bay real-time API.
          Standard Muni buses seat ~83 passengers; light rail vehicles (N, T, K, M) carry ~203.
        - **Peak load factors:** Derived from real-time vehicle observations collected April 7-13,
          2026 via the 511 API (610,573 headway observations across routes 49, 38, 38R). Load
          percentages represent the ratio of observed peak-hour passengers to vehicle capacity.

        **Model assumptions:**
        | Parameter | Value | Source |
        |-----------|-------|--------|
        | Avg. household size | 1.8 persons | Census ACS, SF downtown |
        | Transit mode share | 40% | SFMTA Travel Decision Survey (downtown) |
        | Trips per person per day | 2 (round trip) | Standard planning assumption |
        | Share of daily trips in 4-hr peak | 35% | SFMTA peak-to-base ratio |
        | Avg. routes served per location | 3 | DRD grid network density |

        **Calculation:**
        1. **New peak-hour riders per route** = (units x 1.8 x 0.40 x 2 x 0.35) / (4 hrs x 3 routes)
        2. **New load factor** = (current passengers + new riders per bus) / vehicle capacity x 100
        3. **Units to threshold** = units each route can absorb before reaching 85% load

        **Limitations:**
        - Deterministic capacity model, not a statistical regression.
        - Assumes uniform distribution of new riders across serving routes.
        - Does not account for induced demand, mode shift, or SFMTA service changes.
        - Load factors reflect a single week of observations (April 2026).
        """)

    # ── Transit capacity model ───────────────
    DRD_ROUTES = pd.DataFrame([
        {"route": "38R Geary Rapid",   "daily": 29100, "buses_hr": 5.4, "cap": 83, "load_pct": 72, "serves": "FiDi, Union Square"},
        {"route": "14R Mission Rapid", "daily": 23600, "buses_hr": 5.5, "cap": 83, "load_pct": 70, "serves": "Market St corridor"},
        {"route": "N Judah",           "daily": 34900, "buses_hr": 6.0, "cap": 203, "load_pct": 68, "serves": "Market St subway"},
        {"route": "14 Mission",        "daily": 22200, "buses_hr": 5.0, "cap": 83, "load_pct": 65, "serves": "Market St corridor"},
        {"route": "38 Geary",          "daily": 18400, "buses_hr": 4.8, "cap": 83, "load_pct": 58, "serves": "FiDi, Union Square"},
        {"route": "8 Bayshore",        "daily": 18500, "buses_hr": 4.5, "cap": 83, "load_pct": 55, "serves": "SoMa, East Cut"},
        {"route": "T Third",           "daily": 24600, "buses_hr": 5.0, "cap": 203, "load_pct": 55, "serves": "SoMa, Market St"},
        {"route": "30 Stockton",       "daily": 15000, "buses_hr": 4.0, "cap": 83, "load_pct": 55, "serves": "FiDi, Chinatown"},
        {"route": "45 Union/Stockton", "daily": 10600, "buses_hr": 3.5, "cap": 83, "load_pct": 50, "serves": "FiDi, SoMa"},
        {"route": "M Ocean View",      "daily": 20300, "buses_hr": 5.0, "cap": 203, "load_pct": 50, "serves": "Market St subway"},
        {"route": "12 Folsom/Pacific", "daily": 7700,  "buses_hr": 3.0, "cap": 83, "load_pct": 45, "serves": "SoMa, FiDi"},
        {"route": "K Ingleside",       "daily": 15300, "buses_hr": 4.5, "cap": 203, "load_pct": 45, "serves": "Market St subway"},
    ])

    DRD_ROUTES["cap_hr"] = (DRD_ROUTES["buses_hr"] * DRD_ROUTES["cap"]).astype(int)
    DRD_ROUTES["current_pax"] = (DRD_ROUTES["load_pct"] / 100 * DRD_ROUTES["cap"]).round(0)
    DRD_ROUTES["spare_per_bus"] = (DRD_ROUTES["cap"] - DRD_ROUTES["current_pax"]).astype(int)

    HH_SIZE = 1.8
    TRANSIT_SHARE = 0.40
    TRIPS_PER_DAY = 2
    PEAK_SHARE = 0.35
    N_ROUTES_SERVED = 3

    def units_to_threshold(row, threshold_pct=85):
        target_spare = row["cap"] * (threshold_pct / 100) - row["current_pax"]
        if target_spare <= 0:
            return 0
        trips_per_unit = HH_SIZE * TRANSIT_SHARE * TRIPS_PER_DAY * PEAK_SHARE / (4 * N_ROUTES_SERVED)
        return int(target_spare * row["buses_hr"] / trips_per_unit) if trips_per_unit > 0 else 99999

    DRD_ROUTES["units_to_85"] = DRD_ROUTES.apply(
        lambda r: units_to_threshold(r, 85), axis=1)
    DRD_ROUTES["units_to_100"] = DRD_ROUTES.apply(
        lambda r: units_to_threshold(r, 100), axis=1)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(7, 5))
        sorted_routes = DRD_ROUTES.sort_values("load_pct", ascending=True)
        colors_load = [RED if l >= 70 else AMBER if l >= 60 else BLUE
                       for l in sorted_routes["load_pct"]]
        ax.barh(sorted_routes["route"], sorted_routes["load_pct"],
                color=colors_load, alpha=0.85, height=0.6)
        ax.axvline(85, color=RED, linewidth=2, linestyle="--", alpha=0.7)
        ax.text(86, len(sorted_routes) - 1, "Crowding\nthreshold", fontsize=8,
                color=RED, va="top")
        for i, (_, row) in enumerate(sorted_routes.iterrows()):
            ax.text(row["load_pct"] + 1, i, f"{row['load_pct']}%",
                    va="center", fontsize=9)
        ax.set_xlabel("Peak Load Factor (%)")
        ax.set_title("Current Peak Load: DRD Routes", fontweight="bold")
        ax.set_xlim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sorted_cap = DRD_ROUTES.sort_values("units_to_85", ascending=False)
        ax.barh(sorted_cap["route"], sorted_cap["units_to_85"],
                color=PURPLE, alpha=0.85, height=0.6)
        for i, (_, row) in enumerate(sorted_cap.iterrows()):
            ax.text(row["units_to_85"] + 50, i, f"{row['units_to_85']:,}",
                    va="center", fontsize=9)
        ax.set_xlabel("New Residential Units Before 85% Threshold")
        ax.set_title("Absorption Capacity by Route", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Scenario projections ─────────────────
    st.markdown("### Impact Scenarios: What Happens as Conversions Scale?")

    scenarios = [500, 1000, 1875, 3000, 5000, 10000]
    scenario_labels = ["500 units", "1,000 units",
                       f"~1,875 units\n(1.5M sqft cap)", "3,000 units",
                       "5,000 units", "10,000 units"]

    key_routes = ["38R Geary Rapid", "14R Mission Rapid", "14 Mission",
                  "N Judah", "30 Stockton"]
    scenario_data = {}
    for route_name in key_routes:
        row = DRD_ROUTES[DRD_ROUTES["route"] == route_name].iloc[0]
        loads = []
        for units in scenarios:
            new_residents = units * HH_SIZE
            new_daily = new_residents * TRANSIT_SHARE * TRIPS_PER_DAY
            new_peak_hr = new_daily * PEAK_SHARE / 4 / N_ROUTES_SERVED
            new_per_bus = new_peak_hr / row["buses_hr"]
            new_load = (row["current_pax"] + new_per_bus) / row["cap"] * 100
            loads.append(new_load)
        scenario_data[route_name] = loads

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(scenarios))
    route_colors = {
        "38R Geary Rapid": RED, "14R Mission Rapid": AMBER,
        "14 Mission": BLUE, "N Judah": GREEN, "30 Stockton": PURPLE
    }
    for route_name in key_routes:
        ax.plot(x, scenario_data[route_name], marker="o", linewidth=2.5,
                markersize=7, label=route_name, color=route_colors[route_name])
    ax.axhline(85, color=RED, linewidth=2, linestyle="--", alpha=0.7)
    ax.text(len(scenarios) - 0.5, 86, "Crowding threshold (85%)", fontsize=9,
            color=RED, ha="right")
    ax.axhline(100, color="#6B7280", linewidth=1, linestyle=":", alpha=0.5)
    ax.text(len(scenarios) - 0.5, 101, "Full capacity", fontsize=8,
            color="#6B7280", ha="right")
    ax.axvline(2, color=AMBER, linewidth=2, linestyle="--", alpha=0.5)
    ax.annotate("DRD affordability\nthreshold", xy=(2, ax.get_ylim()[0] + 3),
                fontsize=8, ha="center", color=AMBER, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(scenario_labels, fontsize=9)
    ax.set_ylabel("Peak Load Factor (%)")
    ax.set_title("Projected Transit Load by Conversion Scale", fontweight="bold")
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    ax.set_ylim(40, max(max(v) for v in scenario_data.values()) + 10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    total_units_to_85 = DRD_ROUTES["units_to_85"].sum()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Combined Capacity (12 routes)",
                  f"~{total_units_to_85:,} units",
                  delta="before widespread crowding")
    with col2:
        st.metric("Most Constrained Route",
                  DRD_ROUTES.loc[DRD_ROUTES["units_to_85"].idxmin(), "route"],
                  delta=f"{DRD_ROUTES['units_to_85'].min():,} units to 85%")
    with col3:
        st.metric("DRD Tax Cap",
                  "$1.22B",
                  delta="max lifetime tax increment")

    # ══════════════════════════════════════════════════════
    # PART 3: BEHIND THE POLICY
    # ══════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown("## Part 3: Behind the Policy")
    st.markdown("The DRD is framed as a solution to 'increase housing and decrease vacancy.' "
                "But what's actually driving the economics? And who stands to benefit?")

    st.markdown("---")

    # ── The Market Signal ────────────────────
    st.markdown("### The Market Signal: Why Conversions Make Financial Sense")
    st.markdown("The gap between office and residential revenue tells the economic story "
                "behind the DRD. When we account for the 37% of offices sitting empty, "
                "residential use now generates more revenue per unit of space.")

    px = range(len(price_df))
    office_effective_monthly = price_df["effective_office_rent"] * 800 / 12
    office_asking_monthly = price_df["office_rent_psf"] * 800 / 12
    res_monthly = price_df["residential_1br_rent"]

    col1, col2 = st.columns(2)

    with col1:
        # Effective rent comparison
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(px, office_effective_monthly, color=RED, linewidth=2.5, marker='o',
                markersize=4, label='Office (effective)')
        ax.plot(px, office_asking_monthly, color=RED, linewidth=1.5, linestyle=':',
                alpha=0.5, label='Office (asking)')
        ax.fill_between(px, office_effective_monthly, office_asking_monthly,
                        color=RED, alpha=0.1, label='Lost to vacancy')
        ax.plot(px, res_monthly, color=BLUE, linewidth=2.5, marker='s',
                markersize=4, label='1BR Residential')
        ax.axvline(28, color=GREEN, linewidth=1.5, linestyle='--', alpha=0.5)
        ax.annotate("DRD\nLaunches", xy=(28, ax.get_ylim()[1] * 0.95),
                    fontsize=8, color=GREEN, ha='center', fontweight='bold')
        # Find and annotate crossover
        for i in range(len(price_df)):
            if res_monthly.iloc[i] > office_effective_monthly.iloc[i]:
                ax.annotate("Residential\nsurpasses office",
                            xy=(i, res_monthly.iloc[i]),
                            xytext=(i - 4, res_monthly.iloc[i] + 600),
                            fontsize=8, fontweight='bold', color=BLUE,
                            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5))
                break
        ax.set_ylabel("Monthly Revenue per 800 sqft Unit ($)")
        ax.set_title("Effective Rent: Why Conversions Make Sense", fontweight="bold")
        ax.set_xticks(range(0, len(price_df), 4))
        ax.set_xticklabels(price_df["quarter"].values[::4], rotation=45, fontsize=8)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:,.0f}'))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # Office vacancy over time
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.fill_between(px, price_df["office_vacancy_pct"], color=RED, alpha=0.3)
        ax.plot(px, price_df["office_vacancy_pct"], color=RED, linewidth=2.5,
                marker='o', markersize=4)
        ax.axhline(4.7, color=GREEN, linewidth=1.5, linestyle=':', alpha=0.7)
        ax.text(1, 3.0, "Pre-pandemic low: 4.7%", fontsize=8, color=GREEN, fontweight='bold')
        ax.axvline(28, color=GREEN, linewidth=1.5, linestyle='--', alpha=0.5)
        ax.annotate("DRD\nLaunches", xy=(28, 5), fontsize=8, color=GREEN,
                    ha='center', fontweight='bold')
        ax.set_ylabel("Vacancy Rate (%)")
        ax.set_title("1 in 3 Downtown Offices Sit Empty", fontweight="bold")
        ax.set_xticks(range(0, len(price_df), 4))
        ax.set_xticklabels(price_df["quarter"].values[::4], rotation=45, fontsize=8)
        ax.set_ylim(0, 42)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    latest = price_df.iloc[-1]
    eff_monthly = latest["effective_office_rent"] * 800 / 12
    res_rent = latest["residential_1br_rent"]
    premium = res_rent - eff_monthly
    asking_monthly = latest["office_rent_psf"] * 800 / 12
    vacancy = latest["office_vacancy_pct"]
    st.markdown(
        "**The crossover happened in Q4 2022.** Effective office revenue per 800 sqft unit "
        "is now **\\${:,.0f}/mo** vs **\\${:,.0f}/mo** for a 1BR apartment "
        "-- a **\\${:,.0f}/mo residential premium**. The pink shaded area represents "
        "revenue lost to vacancy: offices *could* charge \\${:,.0f}/mo, but with "
        "{:.1f}% vacancy, a third of that revenue doesn't exist.".format(
            eff_monthly, res_rent, premium, asking_monthly, vacancy
        )
    )

    st.caption("Sources: CBRE, JLL, Cushman & Wakefield market reports; Zumper/Zillow "
               "residential rent data; SF.gov vacancy data. Some quarterly values interpolated.")

    st.markdown("---")

    # ── Who Actually Benefits? ───────────────
    st.markdown("### Who Actually Benefits?")

    st.markdown(
        "SF lost **~66,000 residents** during COVID, yet rents have fully recovered and "
        "surpassed 2019 peaks. Fewer people, higher prices — the housing crisis is driven "
        "by **income competition**, not population growth."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Median SF Rent", "$3,665/mo", "requires $146,600/yr income")
    with col2:
        st.metric("Median Household Income", "$139,801/yr", "-$6,799/yr gap")
    with col3:
        st.metric("DRD Tier 1 Housing", "100% market-rate", "zero affordability requirements")

    st.markdown(
        "The first 1,875 units (1.5M sq ft) will have **no affordable housing requirements** "
        "— serving households earning $140K+. Meanwhile, 80% of the city's 66,000 extremely "
        "low-income households already spend more than a third of their income on rent."
    )
    st.caption("Sources: Census ACS population estimates; Zumper median rent data.")

    st.markdown("---")

    # ── Putting It Together ──────────────────
    st.markdown("### What's Really Driving This?")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            "**The official narrative:** Empty offices hurt the tax base and downtown vitality. "
            "Converting them to housing solves two problems at once -- reducing vacancy and "
            "addressing the housing shortage."
        )
    with col2:
        st.markdown(
            "**What the data shows:** Rents are rising despite population decline. The first "
            "1,875 units will be 100% market-rate. The median household already can't afford "
            "median rent. The primary beneficiaries are property owners (who get tax breaks) "
            "and high-income renters (who get new units)."
        )
    with col3:
        st.markdown(
            "**The unanswered question:** Does the DRD represent genuine housing policy, or "
            "is it a property tax subsidy dressed as a housing program? The $1.22B in diverted "
            "tax revenue could alternatively fund affordable housing, transit, or direct rent "
            "subsidies for the residents who need it most."
        )


# ══════════════════════════════════════════════
# TAB 2: HOSPITALITY TASK FORCE
# ══════════════════════════════════════════════

with tab2:
    st.subheader("Downtown Hospitality Safety Task Force (Feb 2025)")
    st.markdown("In February 2025, Mayor Lurie launched a dedicated police task force "
                "covering Union Square, Moscone Center, and Yerba Buena Gardens -- "
                "officers deployed 20 hours a day, 365 days a year. "
                "Crime went down. Officials celebrated. But that's not the whole story.")

    # ── PART 1: THE PROMISE ─────────────────
    st.markdown("### Part 1: The Promise -- Crime Goes Down Downtown")

    hosp_zone = hosp_monthly[hosp_monthly["zone"] == "Hospitality Zone"].copy()
    rest_sf = hosp_monthly[hosp_monthly["zone"] == "Rest of SF"].copy()

    col1, col2 = st.columns([2.5, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(range(len(hosp_zone)), hosp_zone["total_crimes"].values,
                color=RED, marker="o", linewidth=2.5, markersize=7,
                label="Hospitality Zone")
        ax.plot(range(len(rest_sf)), rest_sf["total_crimes"].values,
                color=BLUE, marker="s", linewidth=2.5, markersize=7,
                label="Rest of SF")

        months = hosp_zone["year_month"].values
        feb_idx = list(months).index("2025-02") if "2025-02" in months else None
        if feb_idx is not None:
            ax.axvline(feb_idx, color="#6B7280", linewidth=2, linestyle="--", alpha=0.7)
            ax.annotate("Task Force (Feb 2025)",
                        xy=(feb_idx, ax.get_ylim()[1] * 0.95),
                        fontsize=10, ha="center", color="#6B7280", fontweight="bold")

        ax.set_xticks(range(len(months)))
        ax.set_xticklabels([m[2:] for m in months], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Total Crimes")
        ax.set_title("Monthly Crime: Hospitality Zone vs Rest of SF", fontweight="bold")
        ax.legend(frameon=True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        if feb_idx is not None:
            pre = hosp_zone.iloc[:feb_idx]
            post = hosp_zone.iloc[feb_idx:]
            pre_avg = pre["total_crimes"].mean()
            post_avg = post["total_crimes"].mean()
            pct_change = ((post_avg - pre_avg) / pre_avg) * 100

            st.metric("Pre-Task Force Avg", f"{pre_avg:,.0f}/mo")
            st.metric("Post-Task Force Avg", f"{post_avg:,.0f}/mo",
                      delta=f"{pct_change:+.1f}%")

            pre_prop = pre["Property"].mean()
            post_prop = post["Property"].mean()
            prop_change = ((post_prop - pre_prop) / pre_prop) * 100
            st.metric("Property Crime", f"{post_prop:,.0f}/mo",
                      delta=f"{prop_change:+.1f}% vs pre")

            pre_viol = pre["Violent"].mean()
            post_viol = post["Violent"].mean()
            viol_change = ((post_viol - pre_viol) / pre_viol) * 100
            st.metric("Violent Crime", f"{post_viol:,.0f}/mo",
                      delta=f"{viol_change:+.1f}% vs pre")

    # ── Officials celebrate ───
    st.markdown("---")
    st.markdown("#### The Official Response")

    statements = pd.read_csv("data/official_statements.csv")
    celebrate = statements[statements["mentions_displacement"] == "No"]

    for _, row in celebrate.iterrows():
        st.markdown(
            f"> \"{row['quote']}\"  \n"
            f"> -- **{row['speaker']}**, {row['event']} ({row['date']}) | *{row['source']}*"
        )

    st.markdown("---")

    # ══════════════════════════════════════════════
    # PART 2: BUT THAT'S NOT THE WHOLE STORY
    # ══════════════════════════════════════════════

    st.markdown("### Part 2: But That's Not the Whole Story")
    st.markdown("Crime didn't disappear -- it moved. The neighborhoods surrounding "
                "the hospitality zone absorbed what downtown pushed out.")

    disp = pd.read_csv("data/displacement_crime.csv")

    # ── Side-by-side: total crimes + drug offenses ───
    col1, col2 = st.columns(2)

    months_d = disp[disp["zone"] == "Hospitality Zone"]["year_month"].values
    feb_idx_d = list(months_d).index("2025-02") if "2025-02" in months_d else None

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for zone, color, marker in [("Hospitality Zone", GREEN, "o"),
                                     ("Mission District", RED, "s"),
                                     ("SoMa (Southern)", AMBER, "^")]:
            zd = disp[disp["zone"] == zone]
            ax.plot(range(len(zd)), zd["total_crimes"].values,
                    color=color, marker=marker, linewidth=2, markersize=5,
                    label=zone, alpha=0.9)
        if feb_idx_d is not None:
            ax.axvline(feb_idx_d, color="#6B7280", linewidth=2, linestyle="--", alpha=0.7)
            ax.annotate("Task Force\n(Feb 2025)",
                        xy=(feb_idx_d, ax.get_ylim()[1] * 0.92),
                        fontsize=9, ha="center", color="#6B7280", fontweight="bold")
        ax.set_xticks(range(0, len(months_d), 3))
        ax.set_xticklabels([months_d[i][2:] for i in range(0, len(months_d), 3)],
                           rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Total Crimes")
        ax.set_title("Total Crime: Downtown vs Surrounding Areas", fontweight="bold")
        ax.legend(frameon=True, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for zone, color, marker in [("Hospitality Zone", GREEN, "o"),
                                     ("Mission District", RED, "s"),
                                     ("SoMa (Southern)", AMBER, "^")]:
            zd = disp[disp["zone"] == zone]
            ax.plot(range(len(zd)), zd["drug_offenses"].values,
                    color=color, marker=marker, linewidth=2, markersize=5,
                    label=zone, alpha=0.9)
        if feb_idx_d is not None:
            ax.axvline(feb_idx_d, color="#6B7280", linewidth=2, linestyle="--", alpha=0.7)
            ax.annotate("Task Force\n(Feb 2025)",
                        xy=(feb_idx_d, ax.get_ylim()[1] * 0.92),
                        fontsize=9, ha="center", color="#6B7280", fontweight="bold")
        ax.set_xticks(range(0, len(months_d), 3))
        ax.set_xticklabels([months_d[i][2:] for i in range(0, len(months_d), 3)],
                           rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Drug Offenses")
        ax.set_title("Drug Offenses: The Displacement Effect", fontweight="bold")
        ax.legend(frameon=True, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Displacement metrics ───
    col1, col2, col3, col4 = st.columns(4)

    mission_pre = disp[(disp["zone"] == "Mission District") & (disp["year_month"] < "2025-02")]
    mission_post = disp[(disp["zone"] == "Mission District") & (disp["year_month"] >= "2025-02")]

    with col1:
        pre_drug = mission_pre["drug_offenses"].mean()
        post_drug = mission_post["drug_offenses"].mean()
        drug_pct = ((post_drug - pre_drug) / pre_drug) * 100
        st.metric("Mission Drug Offenses",
                  f"{post_drug:.0f}/mo",
                  delta=f"{drug_pct:+.0f}% vs pre",
                  delta_color="inverse")
    with col2:
        pre_dc = mission_pre["dispatch_calls"].mean()
        post_dc = mission_post["dispatch_calls"].mean()
        dc_pct = ((post_dc - pre_dc) / pre_dc) * 100
        st.metric("Mission 911 Calls",
                  f"{post_dc:.0f}/mo",
                  delta=f"{dc_pct:+.0f}% vs pre",
                  delta_color="inverse")
    with col3:
        st.metric("16th & Mission Drug Share",
                  "27% of citywide",
                  delta="was 5% pre-task force",
                  delta_color="inverse")
    with col4:
        st.metric("311 Complaints (Mission)",
                  "10-year high",
                  delta="Jan-Feb 2025",
                  delta_color="inverse")

    st.caption("Drug offenses and disorderly conduct are NOT Part 1 crimes -- they are "
               "excluded from the official SFPD dashboards cited in press conferences. "
               "Sources: Mission Local, SF Standard, GrowSF, SFPD DMACC data.")

    st.markdown("---")

    # ── The one time they acknowledged it ───
    st.markdown("#### Did Officials Acknowledge This?")

    col1, col2 = st.columns([1.3, 1])

    with col1:
        yes_count = (statements["mentions_displacement"] == "Yes").sum()
        no_count = (statements["mentions_displacement"] == "No").sum()

        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.barh(["Acknowledges\nDisplacement", "No Mention"],
                       [yes_count, no_count],
                       color=[GREEN, RED], alpha=0.85, height=0.5)
        for bar, val in zip(bars, [yes_count, no_count]):
            ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                    f"{val}", va="center", fontsize=13, fontweight="bold")
        ax.set_xlabel("Number of Public Statements")
        ax.set_title("9 Major Statements Tracked: How Many Mention Displacement?",
                     fontweight="bold")
        ax.set_xlim(0, no_count + 2)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        acknowledge = statements[statements["mentions_displacement"] == "Yes"]
        st.markdown("**The only times displacement was acknowledged:**")
        for _, row in acknowledge.iterrows():
            st.markdown(
                f"> \"{row['quote']}\"  \n"
                f"> -- **{row['speaker']}**, {row['event']} ({row['date']})"
            )
        st.markdown("")
        st.markdown(
            "*Both acknowledgments came only when officials were directly pressed "
            "by journalists or at a Board of Supervisors hearing -- never in proactive "
            "communications.*"
        )

    st.markdown("---")

    # ══════════════════════════════════════════════
    # PUBLIC SENTIMENT: WHAT RESIDENTS ARE SAYING
    # ══════════════════════════════════════════════

    st.markdown("### Public Sentiment: What Residents in SoMa and the Mission Are Saying")
    st.markdown("Officials celebrate downtown wins. These are the voices of residents "
                "living in the neighborhoods absorbing the displacement -- parents, "
                "longtime residents, youth, and community organizers. "
                "Quotes are drawn from Mission Local, SF Standard, ABC7, NBC Bay Area, "
                "Axios SF, and community meetings (2025-2026).")

    sentiment = pd.read_csv("data/resident_sentiment.csv")

    # ── Filter and show quotes ──
    st.markdown("#### Voices from the Neighborhood")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filter_nbhd = st.selectbox("Filter by neighborhood:",
                                    ["All", "Mission", "SoMa"],
                                    key="sentiment_nbhd")
    with col_f2:
        filter_topic = st.selectbox("Filter by topic:",
                                     ["All", "Drugs", "Public Safety",
                                      "Homelessness", "Policy"],
                                     key="sentiment_topic")

    filtered = sentiment.copy()
    if filter_nbhd != "All":
        filtered = filtered[filtered["neighborhood"] == filter_nbhd]
    if filter_topic != "All":
        filtered = filtered[filtered["topic"] == filter_topic]

    if len(filtered) == 0:
        st.info("No quotes match the selected filters.")
    else:
        # Two-column layout for quotes
        col1, col2 = st.columns(2)
        cols = [col1, col2]
        for i, (_, row) in enumerate(filtered.iterrows()):
            with cols[i % 2]:
                sent_emoji = "&#128309;" if row["sentiment"] == "Negative" else \
                             "&#128993;" if row["sentiment"] == "Mixed" else "&#128994;"
                st.markdown(
                    f"<div style='background-color: #F9FAFB; padding: 12px; "
                    f"border-left: 4px solid "
                    f"{RED if row['sentiment'] == 'Negative' else AMBER if row['sentiment'] == 'Mixed' else GREEN}; "
                    f"margin-bottom: 12px; border-radius: 4px;'>"
                    f"<div style='font-style: italic; color: #1F2937; margin-bottom: 8px;'>"
                    f"\"{row['quote']}\"</div>"
                    f"<div style='font-size: 0.85em; color: #6B7280;'>"
                    f"— <b>{row['speaker']}</b> ({row['identity']})<br>"
                    f"<span style='color: #9CA3AF;'>{row['neighborhood']} · {row['topic']} · "
                    f"<a href='{row['url']}' target='_blank' style='color: #6366F1;'>{row['source']}</a> · "
                    f"{row['date']}</span></div></div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")

    # ── Summary metrics ──
    col1, col2, col3 = st.columns(3)
    neg = (sentiment["sentiment"] == "Negative").sum()
    mix = (sentiment["sentiment"] == "Mixed").sum()
    pos = (sentiment["sentiment"] == "Positive").sum()
    total = len(sentiment)

    with col1:
        st.metric("Negative sentiment", f"{neg}/{total}",
                  f"{neg / total * 100:.0f}% of quotes")
    with col2:
        st.metric("Mixed sentiment", f"{mix}/{total}",
                  f"{mix / total * 100:.0f}% of quotes")
    with col3:
        st.metric("Positive sentiment", f"{pos}/{total}",
                  f"{pos / total * 100:.0f}% of quotes")

    st.markdown("")

    # ── Closing insight ──
    st.markdown("#### The Pattern in Residents' Voices")
    st.markdown(
        "**The disconnect:** While the mayor says *\"This city is safe\"* and cites "
        "historic crime lows, residents in SoMa and the Mission describe their "
        "neighborhoods as *\"a shit show every night,\"* *\"like a third world country,\"* "
        "and *\"containment zones for the city's problems.\"* "
        "The narrative gap between **official statements** and **resident experience** "
        "is the story the numbers alone don't tell."
    )


# ══════════════════════════════════════════════
# TAB 3: DATA + VARIABLES
# ══════════════════════════════════════════════

with tab3:
    st.subheader("Data + Variable Definitions")

    # ── Office-to-Residential & DRD ──────────
    st.markdown("### Office-to-Residential & DRD")

    col_data, col_dict = st.columns([1.3, 1])

    with col_data:
        st.markdown("**Transit Capacity Model — Key Variables**")
        st.markdown("""
        | Variable | Definition | Value / Source |
        |----------|-----------|----------------|
        | **Daily Boardings** | Avg weekday boardings per route | SFMTA ridership reports |
        | **Peak Load Factor** | % of vehicle capacity used during peak hours | 511 API observations (Apr 2026) |
        | **Vehicle Capacity** | Max passengers per vehicle (bus: 83, LRV: 203) | SFMTA vehicle specs |
        | **Buses/Hr (Peak)** | Service frequency during AM/PM peak | SFMTA GTFS schedule |
        | **Units to 85%** | New residential units before route hits crowding threshold | Model estimate |
        | **Household Size** | Avg persons per converted unit | 1.8 (Census ACS, downtown SF) |
        | **Transit Mode Share** | % of residents using Muni | 40% (SFMTA Travel Decision Survey) |
        | **Peak Share** | % of daily trips in 4-hr peak window | 35% (SFMTA peak-to-base ratio) |
        """)

    with col_dict:
        st.markdown("**DRD Program Parameters**")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | **Tier 1 threshold** | 0 - 1.5M sq ft (no affordability) |
        | **Tier 2 threshold** | 1.5M - 7M sq ft (state minimums) |
        | **Tier 3 threshold** | 7M+ sq ft (state + local) |
        | **Tax increment share** | ~64.59% of 1% property tax |
        | **Max duration** | 30 years per project |
        | **District cap** | $1.22 billion |
        | **Enrollment deadline** | December 31, 2032 |
        | **Min residential** | 60% of gross floor area |
        """)

    st.markdown("---")

    # ── Hospitality Task Force ───────────────
    st.markdown("### Hospitality Task Force")

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("**Crime & Displacement Data**")
        disp_preview = pd.read_csv("data/displacement_crime.csv")
        st.dataframe(disp_preview.head(10), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Key Variables**")
        st.markdown("""
        | Variable | Definition |
        |----------|-----------|
        | **zone** | Hospitality Zone vs surrounding neighborhoods |
        | **year_month** | Monthly time period |
        | **total_crimes** | All reported crimes in zone |
        | **drug_offenses** | Drug-related offenses (not Part 1 crimes) |
        | **dispatch_calls** | 911 dispatch calls to zone |
        | **treated** | 1 = Hospitality Zone, 0 = control |
        | **post** | 1 = after Feb 2025 task force launch |
        """)

    st.markdown("---")

    # ── Data Sources ─────────────────────────
    st.markdown("### Data Sources")
    st.markdown("""
    | Dataset | Source | Coverage | Used In |
    |---------|--------|----------|---------|
    | Hospitality Zone Crime | SFPD Incident Reports | Monthly, 2024-2025 | Hospitality tab |
    | Displacement Crime | SFPD + 311 Data | Monthly, 2024-2025 | Hospitality tab |
    | Official Statements | Press releases, news | Feb-Jul 2025 | Hospitality tab |
    | Development Pipeline | SF Planning Dept | Historical permits | DRD tab |
    | DRD Program Guidelines | AB 2488 (2024), AB 1445 (2025), Financing Plan (Feb 2026) | Program rules | DRD tab |
    | Muni Ridership | SFMTA Monthly Reports | 2019-present | DRD transit model |
    | Real-time Transit | 511 SF Bay API | Apr 7-13, 2026 (610K obs) | DRD transit model |
    | SFMTA Service Data | GTFS Schedule | Current schedules | DRD transit model |
    | Demographics | Census ACS 5-Year | 2020-2024 | Transit assumptions |
    """)
