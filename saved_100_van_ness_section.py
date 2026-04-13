"""
SAVED: 100 Van Ness Case Study section for app.py
This was removed from tab2 on 2026-04-13 but can be re-inserted.
Place this code inside `with tab2:` before the DRD program section.
Requires: m_van_ness, m_housing, muni_trend, muni_district, muni_race models/data from load_and_fit()
"""

# ══════════════════════════════════════════
# CASE STUDY — 100 VAN NESS
# ══════════════════════════════════════════

#    st.markdown("## Case Study: 100 Van Ness")
#    st.markdown("Converted from office to 399 residential units in 2015 "
#                "(Hayes Valley / Civic Center, BOS District 6).")
#
#    # ── Housing DiD ───────────────────────────
#    pct_effect = (np.exp(m_van_ness.params["van_ness_x_post"]) - 1) * 100
#
#    col1, col2 = st.columns([2, 1])
#    with col1:
#        keep_vn = ["van_ness_treated", "post_van_ness", "van_ness_x_post",
#                    "density", "building_age", "crime_rate"]
#        labels_vn = ["Near 100 Van Ness", "After 2015", "DiD Effect",
#                     "Housing Density", "Building Age", "Crime Rate"]
#        fig = coef_chart(m_van_ness, keep_vn, labels_vn,
#                         "Effect on Property Values", GREEN)
#        st.pyplot(fig)
#        plt.close()
#
#    with col2:
#        st.metric("DiD Effect", f"{m_van_ness.params['van_ness_x_post']:.3f}",
#                  delta="p<0.001")
#        st.metric("In Plain English", f"{abs(pct_effect):.1f}% slower growth")
#        st.metric("Observations", f"{int(m_van_ness.nobs)}")
#
#    st.markdown("---")
#    st.markdown("**What factors drive property values?**")
#
#    keep_h = ["density", "pct_residential", "building_age", "median_stories",
#              "crime_rate", "violent_rate", "log_income"]
#    labels_h = ["Housing Density", "% Residential Land", "Building Age",
#                "Building Height", "Total Crime Rate", "Violent Crime Rate",
#                "Household Income (log)"]
#    fig = coef_chart(m_housing, keep_h, labels_h,
#                     f"Property Value Drivers  |  R-sq = {m_housing.rsquared:.3f}", GREEN)
#    st.pyplot(fig)
#    plt.close()
#
#    st.markdown("---")
#
#    # ── Transit Impact from 100 Van Ness ─────
#    st.markdown("### Transit Reliability: A Confounded Story")
#    st.markdown("100 Van Ness sits on the Van Ness corridor (Route 49) in District 6. "
#                "The SF City Survey tracks how residents rate Muni reliability over time -- "
#                "but a major confounder emerged at the same time as the conversion.")
#
#    st.info("**Important context:** SFMTA launched the "
#            "[Muni Forward initiative](https://www.sfmta.com/press-releases/muni-forward-most-significant-service-improvements-decades) "
#            "on April 25, 2015 -- the same year 100 Van Ness opened. "
#            "Muni Forward increased frequency on 11 routes, extended hours on 7 Express routes, "
#            "and rebranded 6 Limited routes as 'Rapid.' Any reliability gains post-2015 are "
#            "almost certainly driven by this city-wide service overhaul, not the conversion itself.")
#
#    col1, col2 = st.columns([2.5, 1])
#
#    with col1:
#        fig, ax = plt.subplots(figsize=(10, 4.5))
#        ax.plot(muni_trend["year"], muni_trend["reliability"], color=PURPLE,
#                marker="o", linewidth=2.5, markersize=7, label="Muni Reliability")
#        ax.plot(muni_trend["year"], muni_trend["safety"], color=RED,
#                marker="s", linewidth=2, markersize=6, alpha=0.7, label="Muni Safety")
#        ax.axvline(2015, color=GREEN, linewidth=2, linestyle="--", alpha=0.8)
#        ax.annotate("Muni Forward launches\n& 100 Van Ness opens",
#                    xy=(2015, 2.25),
#                    fontsize=9, ha="center", color=GREEN, fontweight="bold")
#        ax.set_xlabel("Year")
#        ax.set_ylabel("Avg Rating (1-5)")
#        ax.set_ylim(2, 4.2)
#        ax.set_title("SF Resident Perceptions of Muni (City Survey)", fontweight="bold")
#        ax.legend(frameon=True, loc="upper left")
#        plt.tight_layout()
#        st.pyplot(fig)
#        plt.close()
#
#    with col2:
#        pre_rel = muni_trend[muni_trend["year"] < 2015]["reliability"].mean()
#        post_rel = muni_trend[muni_trend["year"] >= 2015]["reliability"].mean()
#        rel_change = ((post_rel - pre_rel) / pre_rel) * 100
#        st.metric("Pre-2015 Avg", f"{pre_rel:.2f}/5")
#        st.metric("Post-2015 Avg", f"{post_rel:.2f}/5",
#                  delta=f"{rel_change:+.1f}%")
#        st.caption("Reliability improved city-wide after 2015, but we cannot isolate "
#                   "the effect of 100 Van Ness from Muni Forward's service expansion.")
#
#    st.markdown("---")
#
#    # District comparison
#    st.markdown("### Reliability by Supervisor District")
#    st.markdown("District 6 (SoMa/Tenderloin/Civic Center) is where 100 Van Ness is located. "
#                "How does it compare to neighboring districts?")
#
#    col1, col2 = st.columns(2)
#
#    dist_names = {
#        1: "D1 Richmond", 2: "D2 Marina", 3: "D3 North Beach",
#        4: "D4 Sunset", 5: "D5 Haight/Western Addition",
#        6: "D6 SoMa/Tenderloin", 7: "D7 West of Twin Peaks",
#        8: "D8 Castro/Noe Valley", 9: "D9 Mission",
#        10: "D10 Bayview/Potrero", 11: "D11 Excelsior/OMI"
#    }
#
#    with col1:
#        dist_data = muni_district[muni_district["BOSdistrict"].isin(range(1, 12))].copy()
#        dist_data["period"] = dist_data["year"].apply(
#            lambda y: "Post-2015" if y >= 2015 else "Pre-2015")
#        pre_post = dist_data.groupby(["BOSdistrict", "period"])["reliability"].mean().unstack()
#        pre_post["change"] = pre_post["Post-2015"] - pre_post["Pre-2015"]
#        pre_post = pre_post.sort_values("change", ascending=True)
#
#        fig, ax = plt.subplots(figsize=(7, 5))
#        y_pos = range(len(pre_post))
#        bar_colors = [PURPLE if d == 6 else "#94A3B8" for d in pre_post.index]
#        ax.barh(y_pos, pre_post["change"].values, color=bar_colors, height=0.6, alpha=0.85)
#        ax.set_yticks(list(y_pos))
#        ax.set_yticklabels([dist_names.get(d, f"D{d}") for d in pre_post.index], fontsize=9)
#        ax.axvline(0, color="#6B7280", linewidth=1, linestyle="--", alpha=0.5)
#        for i, val in enumerate(pre_post["change"].values):
#            ax.text(val + 0.02, i, f"{val:+.2f}", va="center", fontsize=9)
#        ax.set_xlabel("Change in Reliability Rating")
#        ax.set_title("Reliability Change: Pre vs Post 2015", fontweight="bold")
#        plt.tight_layout()
#        st.pyplot(fig)
#        plt.close()
#
#    with col2:
#        key_dists = [5, 6, 8, 9]
#        fig, ax = plt.subplots(figsize=(7, 5))
#        colors_d = {5: AMBER, 6: PURPLE, 8: BLUE, 9: RED}
#        for d in key_dists:
#            sub = muni_district[muni_district["BOSdistrict"] == d]
#            ax.plot(sub["year"], sub["reliability"], color=colors_d[d],
#                    marker="o", linewidth=2, markersize=5,
#                    label=dist_names.get(d, f"D{d}"))
#        ax.axvline(2015, color="#6B7280", linewidth=2, linestyle="--", alpha=0.5)
#        ax.set_xlabel("Year")
#        ax.set_ylabel("Reliability Rating (1-5)")
#        ax.set_ylim(2, 4)
#        ax.set_title("Reliability Trend: Districts Near 100 Van Ness", fontweight="bold")
#        ax.legend(frameon=True, fontsize=8)
#        plt.tight_layout()
#        st.pyplot(fig)
#        plt.close()
#
#    st.markdown("---")
#
#    # Muni perceptions by race
#    st.markdown("### Muni Perceptions by Race")
#
#    race_map = {
#        "African American/Black": "Black",
#        "Asian/Pacific Islander": "Asian/PI",
#        "Caucasian/White": "White",
#        "Hispanic/Latino": "Hispanic",
#        "Mixed ethnicity or race": "Mixed",
#    }
#    muni_race_plot = muni_race.copy()
#    muni_race_plot["race"] = muni_race_plot["dem_raceeth"].map(race_map)
#    muni_race_plot = muni_race_plot.dropna(subset=["race"])
#    muni_race_plot = muni_race_plot[muni_race_plot["n"] >= 100].sort_values(
#        "reliability", ascending=True)
#
#    fig, ax = plt.subplots(figsize=(8, 3.5))
#    x = range(len(muni_race_plot))
#    width = 0.35
#    ax.bar([i - width/2 for i in x], muni_race_plot["reliability"].values,
#           width, label="Reliability", color=PURPLE, alpha=0.85)
#    ax.bar([i + width/2 for i in x], muni_race_plot["safety"].values,
#           width, label="Safety on Muni", color=RED, alpha=0.85)
#    ax.set_xticks(list(x))
#    ax.set_xticklabels(muni_race_plot["race"].values, fontsize=10)
#    ax.set_ylabel("Avg Rating (1-5)")
#    ax.set_ylim(2, 4)
#    ax.set_title("Muni Reliability and Safety Perceptions by Race", fontweight="bold")
#    ax.legend(frameon=True)
#    plt.tight_layout()
#    st.pyplot(fig)
#    plt.close()
