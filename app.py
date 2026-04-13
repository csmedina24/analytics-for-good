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

API_KEY_511 = "8f4f06e6-9500-42e4-be5f-464e8bc67641"


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

    return (crime, housing, temescal, temescal_filtered,
            m_total, m_violent, m_housing, m_van_ness, m_tem, m_prop, m_viol,
            survey_race, survey_trend, hosp_demo, hosp_monthly,
            muni_trend, muni_district, muni_race)

(crime_panel, housing_panel, temescal_yr, temescal_filt,
 m_total, m_violent, m_housing, m_van_ness, m_tem, m_prop, m_viol,
 survey_race, survey_trend, hosp_demo, hosp_monthly,
 muni_trend, muni_district, muni_race) = load_and_fit()


# ── 511 API FUNCTIONS ─────────────────────────
@st.cache_data(ttl=120)
def get_muni_routes():
    """Fetch list of all Muni routes from 511 API."""
    url = (f"https://api.511.org/transit/lines?api_key={API_KEY_511}"
           f"&operator_id=SF&format=json")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8-sig")
        data = json.loads(raw)
        routes = {r["Id"]: r["Name"] for r in data}
        return routes
    except Exception:
        return {}


@st.cache_data(ttl=60)
def get_live_arrivals(route_id):
    """Fetch live StopMonitoring data for a specific route."""
    url = (f"https://api.511.org/transit/StopMonitoring?api_key={API_KEY_511}"
           f"&agency=SF&format=json")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8-sig")
        data = json.loads(raw)
        delivery = data.get("ServiceDelivery", {}).get("StopMonitoringDelivery", {})
        if isinstance(delivery, list):
            visits = delivery[0].get("MonitoredStopVisit", []) if delivery else []
        else:
            visits = delivery.get("MonitoredStopVisit", [])

        results = []
        for visit in visits:
            j = visit.get("MonitoredVehicleJourney", {})
            if j.get("LineRef", "") != route_id:
                continue
            call = j.get("MonitoredCall", {})
            loc = j.get("VehicleLocation", {})

            aimed = call.get("AimedArrivalTime", "")
            expected = call.get("ExpectedArrivalTime", "")

            delay = None
            if aimed and expected:
                try:
                    a = datetime.fromisoformat(aimed.replace("Z", "+00:00"))
                    e = datetime.fromisoformat(expected.replace("Z", "+00:00"))
                    delay = int((e - a).total_seconds())
                except Exception:
                    pass

            results.append({
                "stop": call.get("StopPointName", "Unknown"),
                "direction": j.get("DirectionRef", ""),
                "vehicle": j.get("VehicleRef", ""),
                "aimed": aimed,
                "expected": expected,
                "delay_sec": delay,
                "lat": float(loc.get("Latitude", 0) or 0),
                "lon": float(loc.get("Longitude", 0) or 0),
                "occupancy": j.get("Occupancy", ""),
            })
        return results
    except Exception as e:
        return []


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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Upzoning", "Office-to-Residential & DRD",
    "Hospitality Task Force", "Live Muni Tracker", "Data + Variables"
])


# ══════════════════════════════════════════════
# TAB 1: UPZONING
# ══════════════════════════════════════════════

with tab1:
    st.subheader("Upzoning and Densification")
    st.markdown("Oakland upzoned Temescal in 2015. What happened to crime -- and what factors explain it?")

    col_chart, col_metrics = st.columns([2, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plot_data = temescal_filt
        for nbhd, color, marker in [("Temescal", RED, "o"), ("Control", BLUE, "s")]:
            sub = plot_data[plot_data["neighborhood"] == nbhd]
            ax.plot(sub["year"], sub["total_crime"], color=color, marker=marker,
                    linewidth=2.5, markersize=7, label=nbhd)
        ax.axvline(2015, color="#6B7280", linewidth=2, linestyle="--", alpha=0.7)
        ax.annotate("Upzoning (2015)", xy=(2015, ax.get_ylim()[1] * 0.92),
                    fontsize=10, ha="center", color="#6B7280", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Total Crimes")
        ax.legend(frameon=True)
        ax.set_xticks(sorted(plot_data["year"].unique()))
        ax.set_title("Temescal vs Control: Crime Trends (2012-2020)", fontweight="bold")
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
        st.caption("DiD estimates filtered to 2012-2020 to exclude post-COVID spike")

    st.markdown("---")

    with st.expander("Full trend (2012-2023) for context"):
        fig2, ax2 = plt.subplots(figsize=(9, 4))
        for nbhd, color, marker in [("Temescal", RED, "o"), ("Control", BLUE, "s")]:
            sub = temescal_yr[temescal_yr["neighborhood"] == nbhd]
            ax2.plot(sub["year"], sub["total_crime"], color=color, marker=marker,
                     linewidth=2.5, markersize=6, label=nbhd)
        ax2.axvline(2015, color="#6B7280", linewidth=2, linestyle="--", alpha=0.6)
        ax2.axvspan(2021, 2023, alpha=0.08, color=RED)
        ax2.annotate("Post-COVID spike", xy=(2022, ax2.get_ylim()[1] * 0.85),
                     fontsize=9, ha="center", color=RED, fontstyle="italic")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Total Crimes")
        ax2.legend(frameon=True)
        ax2.set_xticks(sorted(temescal_yr["year"].unique()))
        ax2.set_title("Full Trend: Temescal vs Control (2012-2023)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        st.markdown("The sharp divergence after 2020 is likely driven by pandemic-era "
                    "disruptions and recovery effects rather than the 2015 upzoning alone.")

    st.markdown("---")
    st.markdown("**What factors explain this?** The crime regression shows which neighborhood "
                "characteristics drive crime rates:")

    col1, col2 = st.columns(2)
    keep = ["density", "log_median_value", "pct_residential", "log_income"]
    labels = ["Housing Density", "Property Value (log)", "% Residential Land",
              "Household Income (log)"]

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
    st.subheader("Downtown Revitalization Financing District")
    st.markdown("AB 2488 (2024) created a special financing district to incentivize "
                "commercial-to-residential conversions downtown. How will these conversions "
                "impact transit capacity, and what are the equity tradeoffs?")

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

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        The **Downtown Revitalization Financing District** was established on February 12, 2026
        under AB 2488. It finances commercial-to-residential conversions by returning
        a share of the property tax increment generated by the project for up to **30 years**.

        **Key eligibility criteria:**
        - Located within the District boundaries (downtown C-3 zoning districts)
        - Convert existing Commercial space to Residential use
        - At least **60% of gross floor area** must be Residential
        - Must enroll **before** building permit issuance
        - Enrollment deadline: **December 31, 2032**
        """)
    with col2:
        st.markdown("""
        **Affordable housing requirements:**
        - First **1.5M sq ft** of conversions: **no additional** affordability beyond local law
        - After 1.5M sq ft, projects must meet **one of**:
            - 5% of rental units at **50% AMI** (very low income) for 55 years
            - 10% of rental units at **80% AMI** (lower income) for 55 years
            - 10% of for-sale units at **120% AMI** (moderate income) for 45 years
        - Local requirements (Planning Code Sec. 415) waived for C-3 district projects
          up to the first **7M sq ft** of completions
        """)

    st.markdown("---")

    # ── Pipeline in the District ─────────────
    st.markdown("### Eligible Buildings in the District")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Historical Conversions", f"{len(off2res_dedup)}",
                  delta=f"{completed_conv['proposed_units'].sum():.0f} completed units")
    with col2:
        st.metric("Active Pipeline", f"{len(active_conv)}",
                  delta=f"{active_conv['proposed_units'].sum():.0f} units filed/issued")
    with col3:
        est_units_at_threshold = 1_500_000 / 800
        st.metric("1.5M Sq Ft Threshold",
                  f"~{est_units_at_threshold:,.0f} units",
                  delta="zero affordability add-on")
    with col4:
        st.metric("Max Tax Increment", "$1.22B",
                  delta="over district lifetime")

    col1, col2 = st.columns(2)

    with col1:
        nbhd_stats = off2res_dedup.groupby("neighborhoods_analysis_boundaries").agg(
            projects=("address", "count"),
            total_units=("proposed_units", "sum")
        ).sort_values("total_units", ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        short_names = [n.split("/")[0] if "/" in n else n for n in nbhd_stats.index]
        ax.barh(short_names, nbhd_stats["total_units"].values,
                color=GREEN, alpha=0.85, height=0.6)
        for i, (units, projects) in enumerate(
                zip(nbhd_stats["total_units"].values, nbhd_stats["projects"].values)):
            ax.text(units + 30, i, f"{int(units)} units ({projects} projects)",
                    va="center", fontsize=9)
        ax.set_xlabel("Proposed Units")
        ax.set_title("Office-to-Residential Conversions by Neighborhood", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        status_counts = off2res_dedup["status"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        status_colors = {"complete": GREEN, "issued": BLUE, "filed": AMBER,
                         "expired": "#94A3B8", "cancelled": RED, "withdrawn": "#6B7280"}
        colors = [status_colors.get(s, "#999") for s in status_counts.index]
        ax.pie(status_counts.values, labels=status_counts.index, colors=colors,
               autopct="%1.0f%%", startangle=90, textprops={"fontsize": 10})
        ax.set_title("Project Status Distribution", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ══════════════════════════════════════════
    # TRANSIT CAPACITY IMPACT
    # ══════════════════════════════════════════

    st.markdown("### Predicting Transit Impact of DRD Conversions")
    st.markdown("Using current Muni ridership, vehicle capacity, and frequency data, "
                "we model how new residents from conversions would load the transit routes "
                "serving the financing district.")

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
        ax.set_title("Current Peak Load: Financing District Routes", fontweight="bold")
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

    n_tenderloin = len(off2res_dedup[
        off2res_dedup["neighborhoods_analysis_boundaries"] == "Tenderloin"])

    st.warning(
        "**Key equity concerns with the DRD program:**\n\n"
        "1. **First-come-first-served affordability** -- The 1.5M sq ft exemption rewards "
        "speed over community need. Well-resourced developers will likely claim the "
        "exemption first, producing market-rate housing without affordable units.\n\n"
        "2. **Transit burden falls on existing riders** -- Our model shows bus routes hit "
        f"crowding thresholds at ~{DRD_ROUTES['units_to_85'].min():,} units on the most "
        "constrained line. Black and Hispanic riders, who already rate Muni reliability "
        "lowest, would bear the crowding cost of new development.\n\n"
        f"3. **Tenderloin concentration** -- {n_tenderloin} of {len(off2res_dedup)} "
        "historical conversions were in the Tenderloin, where the Hospitality Zone data "
        "shows 42% Asian and 5% Black population. Market-rate housing without affordability "
        "requirements risks displacement.\n\n"
        "4. **Tax increment diverts from transit** -- The program returns up to 64.59% of "
        "property tax increment to developers for 30 years. That's revenue that could fund "
        "the transit capacity these same conversions require.\n\n"
        "5. **No coordination mechanism** -- The DRD has no provisions linking housing "
        "production to transit investment, labor community benefits, or displacement "
        "mitigation beyond minimum affordable housing thresholds."
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
# TAB 3: HOSPITALITY TASK FORCE
# ══════════════════════════════════════════════

with tab3:
    st.subheader("Downtown Hospitality Safety Task Force (Feb 2026)")
    st.markdown("Increased police presence in downtown commercial areas. "
                "What does the data tell us about this zone?")

    # ── Pre/Post Crime Chart ──────────────────
    st.markdown("### Crime Before and After Implementation")

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
        feb_idx = list(months).index("2026-02") if "2026-02" in months else None
        if feb_idx is not None:
            ax.axvline(feb_idx, color="#6B7280", linewidth=2, linestyle="--", alpha=0.7)
            ax.annotate("Task Force (Feb 2026)",
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

    st.markdown("---")

    # ── Polling: Safety Perceptions by Race ───
    st.markdown("### Public Safety Perceptions by Race")
    st.markdown("From the [SF City Survey](https://www.sf.gov/data--city-survey-safety-and-policing) "
                "(2023, n=2,500+).")

    col1, col2 = st.columns(2)

    race_labels = {
        "Asian or Asian American": "Asian",
        "White": "White",
        "Hispanic, Latino, or Spanish Origin": "Hispanic",
        "Black or African American": "Black",
    }
    survey_plot = survey_race.copy()
    survey_plot["race"] = survey_plot["dem_raceeth"].map(race_labels)
    survey_main = survey_plot[survey_plot["n"] >= 100].copy()

    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = range(len(survey_main))
        width = 0.35
        ax.bar([i - width/2 for i in x], survey_main["day_safety"].values,
               width, label="Daytime Safety", color=BLUE, alpha=0.85)
        ax.bar([i + width/2 for i in x], survey_main["night_safety"].values,
               width, label="Nighttime Safety", color=RED, alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(survey_main["race"].values, fontsize=10)
        ax.set_ylabel("Avg Rating (1-5)")
        ax.set_ylim(1, 5)
        ax.set_title("Perceived Safety by Race", fontweight="bold")
        ax.legend(frameon=True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = range(len(survey_main))
        ax.bar([i - width/2 for i in x], survey_main["police_quality"].values,
               width, label="Police Quality", color=GREEN, alpha=0.85)
        ax.bar([i + width/2 for i in x], survey_main["police_trust"].values,
               width, label="Police Trust", color=AMBER, alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(survey_main["race"].values, fontsize=10)
        ax.set_ylabel("Avg Rating (1-5)")
        ax.set_ylim(1, 5)
        ax.set_title("Police Perceptions by Race", fontweight="bold")
        ax.legend(frameon=True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # ── Demographics ──────────────────────────
    st.markdown("### Who Lives in the Hospitality Zone?")
    st.markdown("Racial demographics: hospitality zone vs adjacent neighborhoods (Census ACS).")

    col1, col2 = st.columns(2)

    hosp_d = hosp_demo.copy()
    for zone in hosp_d["zone"].unique():
        mask = hosp_d["zone"] == zone
        total = hosp_d.loc[mask, "estimate"].sum()
        hosp_d.loc[mask, "pct"] = (hosp_d.loc[mask, "estimate"] / total * 100)

    race_colors = {"Asian": BLUE, "White": "#6B7280", "Hispanic": AMBER,
                   "Black": RED, "Other": GREEN}

    with col1:
        hz = hosp_d[hosp_d["zone"] == "Hospitality Zone"].sort_values("pct", ascending=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(hz["race_group"], hz["pct"],
                color=[race_colors.get(r, "#999") for r in hz["race_group"]],
                height=0.6, alpha=0.85)
        for i, (_, row) in enumerate(hz.iterrows()):
            ax.text(row["pct"] + 0.8, i, f"{row['pct']:.1f}%", va="center", fontsize=10)
        ax.set_xlabel("% of Population")
        ax.set_title("Hospitality Zone", fontweight="bold")
        ax.set_xlim(0, 55)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        adj = hosp_d[hosp_d["zone"] == "Adjacent"].sort_values("pct", ascending=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(adj["race_group"], adj["pct"],
                color=[race_colors.get(r, "#999") for r in adj["race_group"]],
                height=0.6, alpha=0.85)
        for i, (_, row) in enumerate(adj.iterrows()):
            ax.text(row["pct"] + 0.8, i, f"{row['pct']:.1f}%", va="center", fontsize=10)
        ax.set_xlabel("% of Population")
        ax.set_title("Adjacent Neighborhoods", fontweight="bold")
        ax.set_xlim(0, 55)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("**Key connections:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Who is policed** -- The hospitality zone is 42% Asian and 15% Hispanic. "
                    "Black residents (5%) are a small share but report the lowest police trust.")
    with col2:
        st.markdown("**Perception gap** -- Asian residents feel the least safe during the day "
                    "despite rating police quality highest.")
    with col3:
        st.markdown("**Early results** -- Crime dropped in the zone post-implementation, "
                    "but it's too early to attribute this to the task force vs seasonal trends.")


# ══════════════════════════════════════════════
# TAB 4: LIVE MUNI TRACKER
# ══════════════════════════════════════════════

with tab4:
    st.subheader("Live Muni Tracker")
    st.markdown("Real-time bus arrival data from the [511 SF Bay API](https://511.org/open-data). "
                "Select a route to see current delays and reliability.")

    routes = get_muni_routes()

    if not routes:
        st.error("Could not load Muni routes from 511 API. Try refreshing.")
    else:
        # Sort routes: numeric first, then alphanumeric
        def route_sort_key(r):
            num = "".join(c for c in r if c.isdigit())
            return (int(num) if num else 999, r)
        sorted_routes = sorted(routes.keys(), key=route_sort_key)

        route_options = [f"{r} - {routes[r]}" for r in sorted_routes]

        # Default to route 49 (Van Ness)
        default_idx = next((i for i, r in enumerate(sorted_routes) if r == "49"), 0)
        selected = st.selectbox("Select a Muni route:", route_options, index=default_idx)
        route_id = selected.split(" - ")[0]

        if st.button("Refresh Data", type="primary"):
            st.cache_data.clear()

        with st.spinner(f"Fetching live data for Route {route_id}..."):
            arrivals = get_live_arrivals(route_id)

        if not arrivals:
            st.warning(f"No active vehicles found for Route {route_id}. "
                       "The line may not be running right now.")
        else:
            df_live = pd.DataFrame(arrivals)

            # Summary metrics
            valid_delays = df_live["delay_sec"].dropna()
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Active Vehicles",
                          f"{df_live['vehicle'].nunique()}")
            with col2:
                st.metric("Stops Served",
                          f"{df_live['stop'].nunique()}")
            with col3:
                if len(valid_delays) > 0:
                    avg_delay = valid_delays.mean()
                    label = "early" if avg_delay < 0 else "late"
                    st.metric("Avg Delay", f"{abs(avg_delay):.0f}s {label}")
                else:
                    st.metric("Avg Delay", "N/A")
            with col4:
                if len(valid_delays) > 0:
                    pct_late = (valid_delays > 0).mean() * 100
                    st.metric("% Running Late", f"{pct_late:.0f}%")
                else:
                    st.metric("% Running Late", "N/A")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                # Delay distribution
                if len(valid_delays) > 0:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    delay_mins = valid_delays / 60
                    ax.hist(delay_mins, bins=30, color=PURPLE, alpha=0.8,
                            edgecolor="white")
                    ax.axvline(0, color=RED, linewidth=2, linestyle="--",
                               label="On time")
                    ax.axvline(delay_mins.mean(), color=AMBER, linewidth=2,
                               linestyle="-", label=f"Avg: {delay_mins.mean():.1f} min")
                    ax.set_xlabel("Delay (minutes) -- negative = early")
                    ax.set_ylabel("Count")
                    ax.set_title(f"Route {route_id}: Current Delay Distribution",
                                 fontweight="bold")
                    ax.legend(frameon=True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

            with col2:
                # Map of vehicle positions
                map_data = df_live[df_live["lat"] != 0].copy()
                if len(map_data) > 0:
                    # Deduplicate by vehicle
                    map_data = map_data.drop_duplicates(subset=["vehicle"])
                    map_data = map_data.rename(columns={"lat": "latitude",
                                                         "lon": "longitude"})
                    st.map(map_data[["latitude", "longitude"]], zoom=12)

            st.markdown("---")

            # Delays by stop
            st.markdown("**Delays by Stop**")
            stop_delays = df_live.dropna(subset=["delay_sec"]).groupby("stop").agg(
                avg_delay=("delay_sec", "mean"),
                count=("delay_sec", "size")
            ).round(0).sort_values("avg_delay", ascending=False)

            stop_delays["avg_delay_min"] = (stop_delays["avg_delay"] / 60).round(1)
            stop_delays["status"] = stop_delays["avg_delay"].apply(
                lambda x: "Late" if x > 60 else "On Time" if x > -60 else "Early"
            )

            display_stops = stop_delays[["avg_delay_min", "count", "status"]].copy()
            display_stops.columns = ["Avg Delay (min)", "Predictions", "Status"]
            st.dataframe(display_stops, use_container_width=True)

            ts = datetime.now().strftime("%I:%M %p")
            st.caption(f"Data as of {ts} | Source: 511 SF Bay API | "
                       "Negative delay = ahead of schedule")


# ══════════════════════════════════════════════
# TAB 5: DATA + VARIABLES
# ══════════════════════════════════════════════

with tab5:
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
    | DRD Program Guidelines | SF DRFD (Feb 2026) | Program rules |
    """)
