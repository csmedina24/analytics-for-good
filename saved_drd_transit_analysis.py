"""
============================================================
DRD Tab — Part 2: Transit Analysis (Saved for Later)
============================================================

This file holds the full "Part 2: The DRD in Practice — Transit Analysis"
section that was previously the middle section of the DRD tab.
Removed from the live dashboard at user request.

The section contained:
- A methodology expander documenting transit-impact assumptions
- A 12-route DRD_ROUTES DataFrame with peak load factors
- Two side-by-side charts: current peak load + absorption capacity
- A scenario projection chart (500–10,000 unit conversion scenarios)
- Three summary metrics

----- HOW TO RESTORE -----

1. In app.py, find the DRD tab block (currently `with tab2:`).
2. Find the marker:
       # ══════════════════════════════════════════════════════
       # PART 3: BEHIND THE POLICY
       # ══════════════════════════════════════════════════════
3. Paste the uncommented block below ABOVE that marker.
4. The original Part 2 also removed the "Part 2:" header — if you
   want this section to become "Part 2" again, the next section
   ("Behind the Policy") needs to be renumbered "Part 3" or kept as
   the new "Part 2" depending on the desired flow.
5. Make sure these globals/constants are still in scope at the top
   of the file: pd, plt, RED, AMBER, BLUE, GREEN, PURPLE, st.

----- WHY IT WAS REMOVED -----

User wanted to keep refining; the transit model relied on stacked
estimates (1.8 persons/HH × 40% mode share × 35% peak share, etc.),
hardcoded peak-load values that were a single-week observation, and
projected impacts for a policy that has produced zero conversions.
The remaining DRD content makes the equity argument cleanly without it.

============================================================
PASTE-IN BLOCK (uncomment to restore as Streamlit content)
============================================================
"""

# # ══════════════════════════════════════════════════════
# # PART 2: THE REALITY CHECK
# # ══════════════════════════════════════════════════════
#
# st.markdown("---")
# st.markdown("## Part 2: The DRD in Practice — Transit Analysis")
# st.markdown("New downtown residents will depend on Muni. "
#             "How much capacity does the system have to absorb them?")
#
# st.markdown("---")
#
# # ── Transit Impact Model ─────────────────
# st.markdown("### Transit Impact: Modeling the Load")
#
# with st.expander("Methodology: How We Derived Transit Impact Predictions", expanded=False):
#     st.markdown("""
#     **Objective:** Estimate how new residential units from DRD conversions would affect
#     peak-hour crowding on Muni routes serving downtown.
#
#     **Data sources:**
#     - **Ridership:** SFMTA monthly ridership reports (2019-present) for daily boarding counts
#       by route.
#     - **Service frequency & capacity:** SFMTA GTFS schedule data and 511 SF Bay real-time API.
#       Standard Muni buses seat ~83 passengers; light rail vehicles (N, T, K, M) carry ~203.
#     - **Peak load factors:** Derived from real-time vehicle observations collected April 7-13,
#       2026 via the 511 API (610,573 headway observations across routes 49, 38, 38R). Load
#       percentages represent the ratio of observed peak-hour passengers to vehicle capacity.
#
#     **Model assumptions:**
#     | Parameter | Value | Source |
#     |-----------|-------|--------|
#     | Avg. household size | 1.8 persons | Census ACS, SF downtown |
#     | Transit mode share | 40% | SFMTA Travel Decision Survey (downtown) |
#     | Trips per person per day | 2 (round trip) | Standard planning assumption |
#     | Share of daily trips in 4-hr peak | 35% | SFMTA peak-to-base ratio |
#     | Avg. routes served per location | 3 | DRD grid network density |
#
#     **Calculation:**
#     1. **New peak-hour riders per route** = (units x 1.8 x 0.40 x 2 x 0.35) / (4 hrs x 3 routes)
#     2. **New load factor** = (current passengers + new riders per bus) / vehicle capacity x 100
#     3. **Units to threshold** = units each route can absorb before reaching 85% load
#
#     **Limitations:**
#     - Deterministic capacity model, not a statistical regression.
#     - Assumes uniform distribution of new riders across serving routes.
#     - Does not account for induced demand, mode shift, or SFMTA service changes.
#     - Load factors reflect a single week of observations (April 2026).
#     """)
#
# # ── Transit capacity model ───────────────
# DRD_ROUTES = pd.DataFrame([
#     {"route": "38R Geary Rapid",   "daily": 29100, "buses_hr": 5.4, "cap": 83, "load_pct": 72, "serves": "FiDi, Union Square"},
#     {"route": "14R Mission Rapid", "daily": 23600, "buses_hr": 5.5, "cap": 83, "load_pct": 70, "serves": "Market St corridor"},
#     {"route": "N Judah",           "daily": 34900, "buses_hr": 6.0, "cap": 203, "load_pct": 68, "serves": "Market St subway"},
#     {"route": "14 Mission",        "daily": 22200, "buses_hr": 5.0, "cap": 83, "load_pct": 65, "serves": "Market St corridor"},
#     {"route": "38 Geary",          "daily": 18400, "buses_hr": 4.8, "cap": 83, "load_pct": 58, "serves": "FiDi, Union Square"},
#     {"route": "8 Bayshore",        "daily": 18500, "buses_hr": 4.5, "cap": 83, "load_pct": 55, "serves": "SoMa, East Cut"},
#     {"route": "T Third",           "daily": 24600, "buses_hr": 5.0, "cap": 203, "load_pct": 55, "serves": "SoMa, Market St"},
#     {"route": "30 Stockton",       "daily": 15000, "buses_hr": 4.0, "cap": 83, "load_pct": 55, "serves": "FiDi, Chinatown"},
#     {"route": "45 Union/Stockton", "daily": 10600, "buses_hr": 3.5, "cap": 83, "load_pct": 50, "serves": "FiDi, SoMa"},
#     {"route": "M Ocean View",      "daily": 20300, "buses_hr": 5.0, "cap": 203, "load_pct": 50, "serves": "Market St subway"},
#     {"route": "12 Folsom/Pacific", "daily": 7700,  "buses_hr": 3.0, "cap": 83, "load_pct": 45, "serves": "SoMa, FiDi"},
#     {"route": "K Ingleside",       "daily": 15300, "buses_hr": 4.5, "cap": 203, "load_pct": 45, "serves": "Market St subway"},
# ])
#
# DRD_ROUTES["cap_hr"] = (DRD_ROUTES["buses_hr"] * DRD_ROUTES["cap"]).astype(int)
# DRD_ROUTES["current_pax"] = (DRD_ROUTES["load_pct"] / 100 * DRD_ROUTES["cap"]).round(0)
# DRD_ROUTES["spare_per_bus"] = (DRD_ROUTES["cap"] - DRD_ROUTES["current_pax"]).astype(int)
#
# HH_SIZE = 1.8
# TRANSIT_SHARE = 0.40
# TRIPS_PER_DAY = 2
# PEAK_SHARE = 0.35
# N_ROUTES_SERVED = 3
#
# def units_to_threshold(row, threshold_pct=85):
#     target_spare = row["cap"] * (threshold_pct / 100) - row["current_pax"]
#     if target_spare <= 0:
#         return 0
#     trips_per_unit = HH_SIZE * TRANSIT_SHARE * TRIPS_PER_DAY * PEAK_SHARE / (4 * N_ROUTES_SERVED)
#     return int(target_spare * row["buses_hr"] / trips_per_unit) if trips_per_unit > 0 else 99999
#
# DRD_ROUTES["units_to_85"] = DRD_ROUTES.apply(
#     lambda r: units_to_threshold(r, 85), axis=1)
# DRD_ROUTES["units_to_100"] = DRD_ROUTES.apply(
#     lambda r: units_to_threshold(r, 100), axis=1)
#
# col1, col2 = st.columns(2)
#
# with col1:
#     fig, ax = plt.subplots(figsize=(7, 5))
#     sorted_routes = DRD_ROUTES.sort_values("load_pct", ascending=True)
#     colors_load = [RED if l >= 70 else AMBER if l >= 60 else BLUE
#                    for l in sorted_routes["load_pct"]]
#     ax.barh(sorted_routes["route"], sorted_routes["load_pct"],
#             color=colors_load, alpha=0.85, height=0.6)
#     ax.axvline(85, color=RED, linewidth=2, linestyle="--", alpha=0.7)
#     ax.text(86, len(sorted_routes) - 1, "Crowding\nthreshold", fontsize=8,
#             color=RED, va="top")
#     for i, (_, row) in enumerate(sorted_routes.iterrows()):
#         ax.text(row["load_pct"] + 1, i, f"{row['load_pct']}%",
#                 va="center", fontsize=9)
#     ax.set_xlabel("Peak Load Factor (%)")
#     ax.set_title("Current Peak Load: DRD Routes", fontweight="bold")
#     ax.set_xlim(0, 100)
#     plt.tight_layout()
#     st.pyplot(fig)
#     plt.close()
#
# with col2:
#     fig, ax = plt.subplots(figsize=(7, 5))
#     sorted_cap = DRD_ROUTES.sort_values("units_to_85", ascending=False)
#     ax.barh(sorted_cap["route"], sorted_cap["units_to_85"],
#             color=PURPLE, alpha=0.85, height=0.6)
#     for i, (_, row) in enumerate(sorted_cap.iterrows()):
#         ax.text(row["units_to_85"] + 50, i, f"{row['units_to_85']:,}",
#                 va="center", fontsize=9)
#     ax.set_xlabel("New Residential Units Before 85% Threshold")
#     ax.set_title("Absorption Capacity by Route", fontweight="bold")
#     plt.tight_layout()
#     st.pyplot(fig)
#     plt.close()
#
# st.markdown("---")
#
# # ── Scenario projections ─────────────────
# st.markdown("### Impact Scenarios: What Happens as Conversions Scale?")
#
# scenarios = [500, 1000, 1875, 3000, 5000, 10000]
# scenario_labels = ["500 units", "1,000 units",
#                    f"~1,875 units\n(1.5M sqft cap)", "3,000 units",
#                    "5,000 units", "10,000 units"]
#
# key_routes = ["38R Geary Rapid", "14R Mission Rapid", "14 Mission",
#               "N Judah", "30 Stockton"]
# scenario_data = {}
# for route_name in key_routes:
#     row = DRD_ROUTES[DRD_ROUTES["route"] == route_name].iloc[0]
#     loads = []
#     for units in scenarios:
#         new_residents = units * HH_SIZE
#         new_daily = new_residents * TRANSIT_SHARE * TRIPS_PER_DAY
#         new_peak_hr = new_daily * PEAK_SHARE / 4 / N_ROUTES_SERVED
#         new_per_bus = new_peak_hr / row["buses_hr"]
#         new_load = (row["current_pax"] + new_per_bus) / row["cap"] * 100
#         loads.append(new_load)
#     scenario_data[route_name] = loads
#
# fig, ax = plt.subplots(figsize=(10, 5))
# x = range(len(scenarios))
# route_colors = {
#     "38R Geary Rapid": RED, "14R Mission Rapid": AMBER,
#     "14 Mission": BLUE, "N Judah": GREEN, "30 Stockton": PURPLE
# }
# for route_name in key_routes:
#     ax.plot(x, scenario_data[route_name], marker="o", linewidth=2.5,
#             markersize=7, label=route_name, color=route_colors[route_name])
# ax.axhline(85, color=RED, linewidth=2, linestyle="--", alpha=0.7)
# ax.text(len(scenarios) - 0.5, 86, "Crowding threshold (85%)", fontsize=9,
#         color=RED, ha="right")
# ax.axhline(100, color="#6B7280", linewidth=1, linestyle=":", alpha=0.5)
# ax.text(len(scenarios) - 0.5, 101, "Full capacity", fontsize=8,
#         color="#6B7280", ha="right")
# ax.axvline(2, color=AMBER, linewidth=2, linestyle="--", alpha=0.5)
# ax.annotate("DRD affordability\nthreshold", xy=(2, ax.get_ylim()[0] + 3),
#             fontsize=8, ha="center", color=AMBER, fontweight="bold")
# ax.set_xticks(list(x))
# ax.set_xticklabels(scenario_labels, fontsize=9)
# ax.set_ylabel("Peak Load Factor (%)")
# ax.set_title("Projected Transit Load by Conversion Scale", fontweight="bold")
# ax.legend(frameon=True, fontsize=8, loc="upper left")
# ax.set_ylim(40, max(max(v) for v in scenario_data.values()) + 10)
# plt.tight_layout()
# st.pyplot(fig)
# plt.close()
#
# total_units_to_85 = DRD_ROUTES["units_to_85"].sum()
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric("Combined Capacity (12 routes)",
#               f"~{total_units_to_85:,} units",
#               delta="before widespread crowding")
# with col2:
#     st.metric("Most Constrained Route",
#               DRD_ROUTES.loc[DRD_ROUTES["units_to_85"].idxmin(), "route"],
#               delta=f"{DRD_ROUTES['units_to_85'].min():,} units to 85%")
# with col3:
#     st.metric("DRD Tax Cap",
#               "$1.22B",
#               delta="max lifetime tax increment")
