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
    "Upzoning", "Office-to-Residential", "Hospitality Task Force",
    "Live Muni Tracker", "Data + Variables"
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
    st.subheader("Office-to-Housing Conversion: 100 Van Ness")
    st.markdown("Converted from office to 399 residential units in 2015 "
                "(Hayes Valley / Civic Center, BOS District 6).")

    # ── Housing DiD ───────────────────────────
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
                  delta="p<0.001")
        st.metric("In Plain English", f"{abs(pct_effect):.1f}% slower growth")
        st.metric("Observations", f"{int(m_van_ness.nobs)}")

    st.markdown("---")
    st.markdown("**What factors drive property values?**")

    keep_h = ["density", "pct_residential", "building_age", "median_stories",
              "crime_rate", "violent_rate", "log_income"]
    labels_h = ["Housing Density", "% Residential Land", "Building Age",
                "Building Height", "Total Crime Rate", "Violent Crime Rate",
                "Household Income (log)"]
    fig = coef_chart(m_housing, keep_h, labels_h,
                     f"Property Value Drivers  |  R-sq = {m_housing.rsquared:.3f}", GREEN)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # ── Transit Impact ────────────────────────
    st.markdown("### Transit Reliability: Did 399 New Residents Strain Muni?")
    st.markdown("100 Van Ness sits on the Van Ness corridor (Route 49) in District 6. "
                "The SF City Survey tracks how residents rate Muni reliability over time.")

    # Muni reliability trend with 2015 line
    col1, col2 = st.columns([2.5, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(muni_trend["year"], muni_trend["reliability"], color=PURPLE,
                marker="o", linewidth=2.5, markersize=7, label="Muni Reliability")
        ax.plot(muni_trend["year"], muni_trend["safety"], color=RED,
                marker="s", linewidth=2, markersize=6, alpha=0.7, label="Muni Safety")
        ax.axvline(2015, color="#6B7280", linewidth=2, linestyle="--", alpha=0.7)
        ax.annotate("100 Van Ness opens (2015)", xy=(2015, ax.get_ylim()[1] * 0.92),
                    fontsize=10, ha="center", color="#6B7280", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Avg Rating (1-5)")
        ax.set_ylim(2, 4.2)
        ax.set_title("SF Resident Perceptions of Muni (City Survey)", fontweight="bold")
        ax.legend(frameon=True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        pre_rel = muni_trend[muni_trend["year"] < 2015]["reliability"].mean()
        post_rel = muni_trend[muni_trend["year"] >= 2015]["reliability"].mean()
        rel_change = ((post_rel - pre_rel) / pre_rel) * 100
        st.metric("Pre-2015 Avg", f"{pre_rel:.2f}/5")
        st.metric("Post-2015 Avg", f"{post_rel:.2f}/5",
                  delta=f"{rel_change:+.1f}%")
        st.caption("Reliability actually improved after 2015, peaking in 2015-2017 "
                   "before declining by 2019")

    st.markdown("---")

    # District comparison
    st.markdown("### Reliability by Supervisor District")
    st.markdown("District 6 (SoMa/Tenderloin/Civic Center) is where 100 Van Ness is located. "
                "How does it compare to neighboring districts?")

    col1, col2 = st.columns(2)

    # Map district numbers to names
    dist_names = {
        1: "D1 Richmond", 2: "D2 Marina", 3: "D3 North Beach",
        4: "D4 Sunset", 5: "D5 Haight/Western Addition",
        6: "D6 SoMa/Tenderloin", 7: "D7 West of Twin Peaks",
        8: "D8 Castro/Noe Valley", 9: "D9 Mission",
        10: "D10 Bayview/Potrero", 11: "D11 Excelsior/OMI"
    }

    with col1:
        # Pre/post comparison by district
        dist_data = muni_district[muni_district["BOSdistrict"].isin(range(1, 12))].copy()
        dist_data["period"] = dist_data["year"].apply(
            lambda y: "Post-2015" if y >= 2015 else "Pre-2015")
        pre_post = dist_data.groupby(["BOSdistrict", "period"])["reliability"].mean().unstack()
        pre_post["change"] = pre_post["Post-2015"] - pre_post["Pre-2015"]
        pre_post = pre_post.sort_values("change", ascending=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        y_pos = range(len(pre_post))
        bar_colors = [PURPLE if d == 6 else "#94A3B8" for d in pre_post.index]
        ax.barh(y_pos, pre_post["change"].values, color=bar_colors, height=0.6, alpha=0.85)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels([dist_names.get(d, f"D{d}") for d in pre_post.index], fontsize=9)
        ax.axvline(0, color="#6B7280", linewidth=1, linestyle="--", alpha=0.5)
        for i, val in enumerate(pre_post["change"].values):
            ax.text(val + 0.02, i, f"{val:+.2f}", va="center", fontsize=9)
        ax.set_xlabel("Change in Reliability Rating")
        ax.set_title("Reliability Change: Pre vs Post 2015", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # Time series for key districts
        key_dists = [5, 6, 8, 9]
        fig, ax = plt.subplots(figsize=(7, 5))
        colors_d = {5: AMBER, 6: PURPLE, 8: BLUE, 9: RED}
        for d in key_dists:
            sub = muni_district[muni_district["BOSdistrict"] == d]
            ax.plot(sub["year"], sub["reliability"], color=colors_d[d],
                    marker="o", linewidth=2, markersize=5,
                    label=dist_names.get(d, f"D{d}"))
        ax.axvline(2015, color="#6B7280", linewidth=2, linestyle="--", alpha=0.5)
        ax.set_xlabel("Year")
        ax.set_ylabel("Reliability Rating (1-5)")
        ax.set_ylim(2, 4)
        ax.set_title("Reliability Trend: Districts Near 100 Van Ness", fontweight="bold")
        ax.legend(frameon=True, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Muni perceptions by race
    st.markdown("### Muni Perceptions by Race")

    race_map = {
        "African American/Black": "Black",
        "Asian/Pacific Islander": "Asian/PI",
        "Caucasian/White": "White",
        "Hispanic/Latino": "Hispanic",
        "Mixed ethnicity or race": "Mixed",
    }
    muni_race_plot = muni_race.copy()
    muni_race_plot["race"] = muni_race_plot["dem_raceeth"].map(race_map)
    muni_race_plot = muni_race_plot.dropna(subset=["race"])
    muni_race_plot = muni_race_plot[muni_race_plot["n"] >= 100].sort_values(
        "reliability", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = range(len(muni_race_plot))
    width = 0.35
    ax.bar([i - width/2 for i in x], muni_race_plot["reliability"].values,
           width, label="Reliability", color=PURPLE, alpha=0.85)
    ax.bar([i + width/2 for i in x], muni_race_plot["safety"].values,
           width, label="Safety on Muni", color=RED, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(muni_race_plot["race"].values, fontsize=10)
    ax.set_ylabel("Avg Rating (1-5)")
    ax.set_ylim(2, 4)
    ax.set_title("Muni Reliability and Safety Perceptions by Race", fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("**Key takeaways:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**No transit strain from conversion** -- Muni reliability perceptions "
                    "*improved* city-wide after 2015. District 6 (100 Van Ness) saw one of "
                    "the largest gains, suggesting the conversion didn't degrade transit service.")
    with col2:
        st.markdown("**Geographic gradient** -- Southern/eastern districts (D8, D9, D10) "
                    "consistently rate Muni reliability lower than northern/western ones, "
                    "reflecting uneven service quality across the city.")


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
    """)
