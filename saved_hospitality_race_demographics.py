# ============================================================
# Hospitality Task Force — Race Perceptions & Demographics
# Removed 2026-04-14 at user's request
# Includes: Part 3 (Who Benefits?), Safety Perceptions by Race, Who Lives in the Hospitality Zone
# To re-insert: paste back into tab2 (Hospitality) before the Data + Variables tab
# Data needed: survey_race, hosp_demo (loaded via load_and_fit())
# ============================================================

#     # ══════════════════════════════════════════════
#     # PART 3: WHO BENEFITS?
#     # ══════════════════════════════════════════════
#
#     st.markdown("### Part 3: Who Benefits?")
#     st.markdown("Visitors and voters see a safer downtown. Residents in surrounding "
#                 "neighborhoods experience the displacement firsthand.")
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         st.markdown("#### The Downtown View")
#         fig, ax = plt.subplots(figsize=(7, 4))
#         visitor_data = [
#             ("Support downtown\npolice (voters)", 73, GREEN),
#             ("Feel safer with\nvisible patrols", 85, GREEN),
#             ("City on right\ntrack (mid-2025)", 52, BLUE),
#         ]
#         labels = [d[0] for d in visitor_data]
#         vals = [d[1] for d in visitor_data]
#         colors = [d[2] for d in visitor_data]
#         bars = ax.barh(labels, vals, color=colors, alpha=0.85, height=0.55)
#         for bar, val in zip(bars, vals):
#             ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
#                     f"{val}%", va="center", fontsize=11, fontweight="bold")
#         ax.set_xlim(0, 100)
#         ax.set_xlabel("% Agreement")
#         ax.set_title("Visitors & Voters: The Task Force Works", fontweight="bold")
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()
#         st.caption("Sources: GrowSF Pulse Poll (Feb 2025, n=423; Jul 2025), "
#                    "NBC Bay Area visitor interviews.")
#
#     with col2:
#         st.markdown("#### The Neighborhood View")
#         fig, ax = plt.subplots(figsize=(7, 4))
#         resident_data = [
#             ("Tenderloin: feel safe\nwalking (day)", 24, RED),
#             ("FiDi: feel safe\nwalking (day)", 23, RED),
#             ("Mission: concerned\nabout displacement", 78, RED),
#         ]
#         labels = [d[0] for d in resident_data]
#         vals = [d[1] for d in resident_data]
#         colors = [d[2] for d in resident_data]
#         bars = ax.barh(labels, vals, color=colors, alpha=0.85, height=0.55)
#         for bar, val in zip(bars, vals):
#             ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
#                     f"{val}%", va="center", fontsize=11, fontweight="bold")
#         ax.set_xlim(0, 100)
#         ax.set_xlabel("% Agreement")
#         ax.set_title("Surrounding Residents: A Different Reality", fontweight="bold")
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()
#         st.caption("Sources: SF City Survey (2023, n=2,500+), Mission Local, "
#                    "community meetings (May 2025).")
#
#     st.markdown("---")
#
#     # ── Polling: Safety Perceptions by Race ───
#     st.markdown("### Public Safety Perceptions by Race")
#     st.markdown("From the [SF City Survey](https://www.sf.gov/data--city-survey-safety-and-policing) "
#                 "(2023, n=2,500+).")
#
#     col1, col2 = st.columns(2)
#
#     race_labels = {
#         "Asian or Asian American": "Asian",
#         "White": "White",
#         "Hispanic, Latino, or Spanish Origin": "Hispanic",
#         "Black or African American": "Black",
#     }
#     survey_plot = survey_race.copy()
#     survey_plot["race"] = survey_plot["dem_raceeth"].map(race_labels)
#     survey_main = survey_plot[survey_plot["n"] >= 100].copy()
#
#     with col1:
#         fig, ax = plt.subplots(figsize=(7, 4))
#         x = range(len(survey_main))
#         width = 0.35
#         ax.bar([i - width/2 for i in x], survey_main["day_safety"].values,
#                width, label="Daytime Safety", color=BLUE, alpha=0.85)
#         ax.bar([i + width/2 for i in x], survey_main["night_safety"].values,
#                width, label="Nighttime Safety", color=RED, alpha=0.85)
#         ax.set_xticks(list(x))
#         ax.set_xticklabels(survey_main["race"].values, fontsize=10)
#         ax.set_ylabel("Avg Rating (1-5)")
#         ax.set_ylim(1, 5)
#         ax.set_title("Perceived Safety by Race", fontweight="bold")
#         ax.legend(frameon=True)
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()
#
#     with col2:
#         fig, ax = plt.subplots(figsize=(7, 4))
#         x = range(len(survey_main))
#         ax.bar([i - width/2 for i in x], survey_main["police_quality"].values,
#                width, label="Police Quality", color=GREEN, alpha=0.85)
#         ax.bar([i + width/2 for i in x], survey_main["police_trust"].values,
#                width, label="Police Trust", color=AMBER, alpha=0.85)
#         ax.set_xticks(list(x))
#         ax.set_xticklabels(survey_main["race"].values, fontsize=10)
#         ax.set_ylabel("Avg Rating (1-5)")
#         ax.set_ylim(1, 5)
#         ax.set_title("Police Perceptions by Race", fontweight="bold")
#         ax.legend(frameon=True)
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()
#
#     st.markdown("---")
#
#     # ── Demographics ──────────────────────────
#     st.markdown("### Who Lives in the Hospitality Zone?")
#     st.markdown("Racial demographics: hospitality zone vs adjacent neighborhoods (Census ACS).")
#
#     col1, col2 = st.columns(2)
#
#     hosp_d = hosp_demo.copy()
#     for zone in hosp_d["zone"].unique():
#         mask = hosp_d["zone"] == zone
#         total = hosp_d.loc[mask, "estimate"].sum()
#         hosp_d.loc[mask, "pct"] = (hosp_d.loc[mask, "estimate"] / total * 100)
#
#     race_colors = {"Asian": BLUE, "White": "#6B7280", "Hispanic": AMBER,
#                    "Black": RED, "Other": GREEN}
#
#     with col1:
#         hz = hosp_d[hosp_d["zone"] == "Hospitality Zone"].sort_values("pct", ascending=True)
#         fig, ax = plt.subplots(figsize=(6, 3.5))
#         ax.barh(hz["race_group"], hz["pct"],
#                 color=[race_colors.get(r, "#999") for r in hz["race_group"]],
#                 height=0.6, alpha=0.85)
#         for i, (_, row) in enumerate(hz.iterrows()):
#             ax.text(row["pct"] + 0.8, i, f"{row['pct']:.1f}%", va="center", fontsize=10)
#         ax.set_xlabel("% of Population")
#         ax.set_title("Hospitality Zone", fontweight="bold")
#         ax.set_xlim(0, 55)
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()
#
#     with col2:
#         adj = hosp_d[hosp_d["zone"] == "Adjacent"].sort_values("pct", ascending=True)
#         fig, ax = plt.subplots(figsize=(6, 3.5))
#         ax.barh(adj["race_group"], adj["pct"],
#                 color=[race_colors.get(r, "#999") for r in adj["race_group"]],
#                 height=0.6, alpha=0.85)
#         for i, (_, row) in enumerate(adj.iterrows()):
#             ax.text(row["pct"] + 0.8, i, f"{row['pct']:.1f}%", va="center", fontsize=10)
#         ax.set_xlabel("% of Population")
#         ax.set_title("Adjacent Neighborhoods", fontweight="bold")
#         ax.set_xlim(0, 55)
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()
#
#     st.markdown("---")
#     st.markdown("**Key connections:**")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.markdown("**Who is policed** -- The hospitality zone is 42% Asian and 15% Hispanic. "
#                     "Black residents (5%) are a small share but report the lowest police trust.")
#     with col2:
#         st.markdown("**Perception gap** -- Asian residents feel the least safe during the day "
#                     "despite rating police quality highest.")
#     with col3:
#         st.markdown("**Who bears the cost** -- The neighborhoods absorbing displaced crime "
#                     "are disproportionately lower-income communities of color.")
