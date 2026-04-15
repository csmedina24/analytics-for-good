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
    st.markdown("AB 2488 (2024) created the Downtown Revitalization Financing District (DRD) "
                "to incentivize commercial-to-residential conversions downtown. How will these "
                "conversions impact transit capacity, and what are the equity tradeoffs?")

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

    # ══════════════════════════════════════════
    # DRD PROGRAM OVERVIEW
    # ══════════════════════════════════════════

    # ── DRD Program Overview ─────────────────
    st.markdown("### Program Overview")

    st.markdown("""
    The **DRD** was established on February 12, 2026 under AB 2488. It finances
    commercial-to-residential conversions by returning a share of the property tax
    increment generated by the project for up to **30 years**.
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
        st.markdown("""
        **Financial structure:**
        - Annual distribution = City's share (~64.59%) of 1% property tax on incremental value
        - Capped at 1/30th of total Qualified Development Costs per year
        - Maximum **30 years** of distributions
        - Total district cap: **$1.22 billion** in allocated tax revenue
        - Admin costs capped at **5%** of tax revenues
        """)

    st.markdown("---")

    # ── Affordability Tiers ──────────────────
    st.markdown("### Affordable Housing Requirements: Three Tiers")
    st.markdown("The DRD creates a phased affordability structure. Projects that enroll "
                "early face fewer requirements -- creating a first-come-first-served race.")

    # Stacked bar showing the three tiers
    fig, ax = plt.subplots(figsize=(10, 2.5))

    # Tier segments (in millions of sq ft)
    tier1_end = 1.5
    tier2_end = 7.0
    tier3_end = 10.0  # illustrative max

    ax.barh(0, tier1_end, left=0, height=0.5, color=GREEN, alpha=0.85,
            edgecolor="white", linewidth=2)
    ax.barh(0, tier2_end - tier1_end, left=tier1_end, height=0.5, color=AMBER,
            alpha=0.85, edgecolor="white", linewidth=2)
    ax.barh(0, tier3_end - tier2_end, left=tier2_end, height=0.5, color=RED,
            alpha=0.85, edgecolor="white", linewidth=2)

    # Labels inside bars
    ax.text(tier1_end / 2, 0, "Tier 1\nNo additional\naffordability",
            ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(tier1_end + (tier2_end - tier1_end) / 2, 0,
            "Tier 2\nState minimums only\n(5-10% affordable)",
            ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.text(tier2_end + (tier3_end - tier2_end) / 2, 0,
            "Tier 3\nState + local\ninclusionary",
            ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    # Threshold markers
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

    # Tier details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Tier 1: 0 - 1.5M sq ft**")
        st.markdown("No state affordability requirements. Local inclusionary (Sec. 415) "
                    "also waived in C-3 districts. These projects can be **100% market-rate**.")
    with col2:
        st.markdown("**Tier 2: 1.5M - 7M sq ft**")
        st.markdown("State minimums kick in -- one of:\n"
                    "- 5% of rental units at **50% AMI** (55 yrs)\n"
                    "- 10% of rental units at **80% AMI** (55 yrs)\n"
                    "- 10% of for-sale units at **120% AMI** (45 yrs)\n\n"
                    "Local inclusionary still waived.")
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

    st.markdown("---")

    # ── Where Could Conversions Happen? ──────
    st.markdown("### Where Could Conversions Happen?")
    st.markdown("The permit pipeline shows buildings with office use in the DRD neighborhoods. "
                "These represent the pool of *potentially eligible* buildings -- not conversions "
                "that have occurred.")

    # Count all office buildings in district (not just ones with residential permits)
    office_buildings = dt_pipe[
        dt_pipe["existing_use"].str.contains("office", case=False, na=False)
    ].copy()
    office_buildings["address"] = (office_buildings["street_number"].astype(str) + " "
                                   + office_buildings["street_name"])
    office_dedup = office_buildings.drop_duplicates("address", keep="first")

    col1, col2 = st.columns(2)

    with col1:
        office_by_nbhd = office_dedup.groupby("neighborhoods_analysis_boundaries").agg(
            buildings=("address", "count"),
            total_units=("proposed_units", "sum")
        ).sort_values("buildings", ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        short_names = [n.split("/")[0] if "/" in n else n for n in office_by_nbhd.index]
        ax.barh(short_names, office_by_nbhd["buildings"].values,
                color=PURPLE, alpha=0.85, height=0.6)
        for i, count in enumerate(office_by_nbhd["buildings"].values):
            ax.text(count + 5, i, f"{count}", va="center", fontsize=9)
        ax.set_xlabel("Office Buildings (unique addresses)")
        ax.set_title("Potential DRD-Eligible Office Buildings", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # Projects that have filed/issued permits for office-to-residential
        if len(active_conv) > 0:
            st.markdown("**Active conversion applications:**")
            for _, row in active_conv.iterrows():
                st.markdown(f"- **{row['address']}** -- {int(row['proposed_units'])} units "
                            f"({row['status']}, {row['neighborhoods_analysis_boundaries']})")
        else:
            st.markdown("**No active conversion applications in the pipeline.**")

        st.markdown("")
        st.info("The DRD program's tax increment incentive is designed to make "
                "conversions financially viable for the first time. The number and "
                "scale of future projects will depend on developer enrollment "
                "before the December 31, 2032 deadline.")

    st.markdown("---")

    # ══════════════════════════════════════════
    # TRANSIT CAPACITY IMPACT
    # ══════════════════════════════════════════

    st.markdown("### Predicting Transit Impact of DRD Conversions")
    st.markdown("Using current Muni ridership, vehicle capacity, and frequency data, "
                "we model how new residents from conversions would load the transit routes "
                "serving the DRD.")

    # ── Transit capacity model ───────────────
    # Route data: name, daily boardings, buses/hr peak, capacity/bus
    # From earlier analysis using 511 API + SFMTA GTFS data
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

    # Calculate spare capacity and units-to-threshold
    DRD_ROUTES["cap_hr"] = (DRD_ROUTES["buses_hr"] * DRD_ROUTES["cap"]).astype(int)
    DRD_ROUTES["current_pax"] = (DRD_ROUTES["load_pct"] / 100 * DRD_ROUTES["cap"]).round(0)
    DRD_ROUTES["spare_per_bus"] = (DRD_ROUTES["cap"] - DRD_ROUTES["current_pax"]).astype(int)

    # Model: avg HH size 1.8, 40% transit mode share, 2 trips/day, 35% in peak 4 hrs
    HH_SIZE = 1.8
    TRANSIT_SHARE = 0.40
    TRIPS_PER_DAY = 2
    PEAK_SHARE = 0.35
    N_ROUTES_SERVED = 3  # avg routes near a conversion

    def units_to_threshold(row, threshold_pct=85):
        spare_hr = row["spare_per_bus"] * row["buses_hr"]
        target_spare = row["cap"] * (threshold_pct / 100) - row["current_pax"]
        if target_spare <= 0:
            return 0
        trips_per_unit = HH_SIZE * TRANSIT_SHARE * TRIPS_PER_DAY * PEAK_SHARE / (4 * N_ROUTES_SERVED)
        return int(target_spare * row["buses_hr"] / trips_per_unit) if trips_per_unit > 0 else 99999

    DRD_ROUTES["units_to_85"] = DRD_ROUTES.apply(
        lambda r: units_to_threshold(r, 85), axis=1)
    DRD_ROUTES["units_to_100"] = DRD_ROUTES.apply(
        lambda r: units_to_threshold(r, 100), axis=1)

    # ── Current load chart ───────────────────
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

    # ── Conversion scenario table ────────────
    st.markdown("### Impact Scenarios: What Happens as Conversions Scale?")
    st.markdown("Each scenario adds new residents to the district. We estimate their transit "
                "demand and show how it loads onto the most constrained routes.")

    scenarios = [500, 1000, 1875, 3000, 5000, 10000]
    scenario_labels = ["500 units", "1,000 units",
                       f"~1,875 units\n(1.5M sqft cap)", "3,000 units",
                       "5,000 units", "10,000 units"]

    # Calculate load impact for key routes
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
    # Mark the 1.5M sqft threshold
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

    st.markdown("---")

    # ── Combined capacity estimate ───────────
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

    st.markdown("---")

    # ══════════════════════════════════════════
    # MUNI BUDGET CRISIS + EQUITY
    # ══════════════════════════════════════════

    st.markdown("### The Transit Funding Collision")
    st.markdown("The DRD program assumes downtown is transit-rich. But SFMTA is facing "
                "historic budget cuts that could undermine that assumption.")

    st.warning("**Budget crisis reverses the trend.** SFMTA faces a **$307 million deficit** "
               "starting July 2026. In **summer 2025**, the agency cut service to 5 lines "
               "(including the 21 Hayes and shortening the 31 Balboa). If two pending tax "
               "ballot measures fail, **up to 20 more lines could be eliminated by September 2027**, "
               "with regular service ending at 9 PM and frequency reductions on surviving routes. "
               "([SF Standard, Mar 2026](https://sfstandard.com/2026/03/29/muni-could-cut-20-lines-next-year-if-ballot-measures-don-t-pass-riders-have-idea/))")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Timeline")
        st.markdown("""
        - **2015**: Muni Forward launches. 100 Van Ness opens. Reliability ratings peak.
        - **2015-2019**: Ratings stay elevated above historical average.
        - **2020-2021**: COVID disrupts ridership and service.
        - **Summer 2025**: SFMTA cuts 5 bus lines due to budget shortfall.
        - **Feb 2026**: DRD program launches, assumes transit-rich downtown.
        - **Nov 2026**: Voters decide on two tax measures to fund Muni.
        - **Sep 2027**: If measures fail, up to 20 additional lines eliminated.
        """)

    with col2:
        st.markdown("#### The Contradiction")
        st.markdown("""
        - The DRD program incentivizes **thousands of new residents** downtown.
        - SFMTA is **cutting the routes** those residents would depend on.
        - Route 38R Geary Rapid is already at **72% load** -- the most constrained
          bus route in the district, with capacity for only
          **{:,} more units** before crowding.
        - **Black and Hispanic riders** depend on Muni at higher rates and
          already rate reliability lowest -- new crowding would
          disproportionately affect them.
        - The DRD program has **no mechanism** to coordinate with SFMTA
          or fund transit capacity.
        """.format(DRD_ROUTES.loc[DRD_ROUTES["route"] == "38R Geary Rapid",
                                   "units_to_85"].iloc[0]))

    st.markdown("---")

    # ── Equity Assessment ────────────────────
    st.markdown("### Equity Assessment")

    st.warning(
        "**Key equity concerns with the DRD program:**\n\n"
        "1. **First-come-first-served affordability** -- The 1.5M sq ft exemption rewards "
        "speed over community need. Well-resourced developers will likely claim the "
        "exemption first, producing market-rate housing without affordable units.\n\n"
        "2. **Transit burden falls on existing riders** -- Our model shows bus routes hit "
        f"crowding thresholds at ~{DRD_ROUTES['units_to_85'].min():,} units on the most "
        "constrained line. Black and Hispanic riders, who already rate Muni reliability "
        "lowest, would bear the crowding cost of new development.\n\n"
        "3. **Tenderloin and SoMa concentration** -- Most eligible office buildings are in "
        "the Tenderloin and Financial District, where the Hospitality Zone data shows 42% "
        "Asian and 5% Black population. Market-rate housing without affordability "
        "requirements risks displacement of existing residents.\n\n"
        "4. **Tax increment diverts from transit** -- The program returns up to 64.59% of "
        "property tax increment to developers for 30 years. That's revenue that could fund "
        "the transit capacity these same conversions require.\n\n"
        "5. **No coordination mechanism** -- The DRD has no provisions linking housing "
        "production to transit investment or displacement mitigation beyond minimum "
        "affordable housing thresholds."
    )

    st.markdown("---")

    # ── Key Takeaways ────────────────────────
    st.markdown("### Key Takeaways")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Scale matters** -- A single conversion may not strain transit, "
                    "but the DRD could enable thousands of units on already-loaded corridors.")
    with col2:
        st.markdown("**Transit is the bottleneck** -- Bus routes like 38R Geary are already "
                    f"at 72% capacity with only ~{DRD_ROUTES.loc[DRD_ROUTES['route']=='38R Geary Rapid', 'units_to_85'].iloc[0]:,} "
                    "units of headroom. The program creates housing demand without transit supply.")
    with col3:
        st.markdown("**Timing is everything** -- The DRD launches just as Muni faces its worst "
                    "funding crisis. If ballot measures fail, the 'transit-rich' downtown that "
                    "justifies the program may no longer exist by 2027.")


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



# ══════════════════════════════════════════════
# TAB 3: DATA + VARIABLES
# ══════════════════════════════════════════════

with tab3:
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
    | City Survey | SF City Survey | 1996-2023 |
    | Transit | 511 SF Bay API | Real-time |
    | Development Pipeline | SF Planning Dept | Historical |
    | DRD Program Guidelines | SF DRD (Feb 2026) | Program rules |
    """)
