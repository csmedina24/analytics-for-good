# ============================================================
# Upzoning Tab — Saved 2026-04-14
# Removed at user's request to focus on Hospitality + Office-to-Residential
# To re-insert: add "Upzoning" back to st.tabs() and paste this as a tab block
# Models needed: m_tem, m_prop, m_viol, m_total, m_violent (from load_and_fit())
# Data needed: temescal_filt, temescal_yr
# Also uses: coef_chart() helper function (still defined in app.py)
# ============================================================

# with tab_upzoning:
#     st.subheader("Upzoning and Densification")
#     st.markdown("Oakland upzoned Temescal in 2015. What happened to crime -- and what factors explain it?")
#
#     col_chart, col_metrics = st.columns([2, 1])
#
#     with col_chart:
#         fig, ax = plt.subplots(figsize=(9, 4.5))
#         plot_data = temescal_filt
#         for nbhd, color, marker in [("Temescal", RED, "o"), ("Control", BLUE, "s")]:
#             sub = plot_data[plot_data["neighborhood"] == nbhd]
#             ax.plot(sub["year"], sub["total_crime"], color=color, marker=marker,
#                     linewidth=2.5, markersize=7, label=nbhd)
#         ax.axvline(2015, color="#6B7280", linewidth=2, linestyle="--", alpha=0.7)
#         ax.annotate("Upzoning (2015)", xy=(2015, ax.get_ylim()[1] * 0.92),
#                     fontsize=10, ha="center", color="#6B7280", fontweight="bold")
#         ax.set_xlabel("Year")
#         ax.set_ylabel("Total Crimes")
#         ax.legend(frameon=True)
#         ax.set_xticks(sorted(plot_data["year"].unique()))
#         ax.set_title("Temescal vs Control: Crime Trends (2012-2020)", fontweight="bold")
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()
#
#     with col_metrics:
#         st.metric("Total Crime", f"+{m_tem.params['treated_x_post']:.0f}/yr",
#                   delta=f"p={m_tem.pvalues['treated_x_post']:.3f}")
#         st.metric("Property Crime", f"+{m_prop.params['treated_x_post']:.0f}/yr",
#                   delta=f"p={m_prop.pvalues['treated_x_post']:.3f}")
#         st.metric("Violent Crime", f"+{m_viol.params['treated_x_post']:.0f}/yr",
#                   delta=f"p={m_viol.pvalues['treated_x_post']:.2f} (n.s.)")
#         st.caption("DiD estimates filtered to 2012-2020 to exclude post-COVID spike")
#
#     st.markdown("---")
#
#     with st.expander("Full trend (2012-2023) for context"):
#         fig2, ax2 = plt.subplots(figsize=(9, 4))
#         for nbhd, color, marker in [("Temescal", RED, "o"), ("Control", BLUE, "s")]:
#             sub = temescal_yr[temescal_yr["neighborhood"] == nbhd]
#             ax2.plot(sub["year"], sub["total_crime"], color=color, marker=marker,
#                      linewidth=2.5, markersize=6, label=nbhd)
#         ax2.axvline(2015, color="#6B7280", linewidth=2, linestyle="--", alpha=0.6)
#         ax2.axvspan(2021, 2023, alpha=0.08, color=RED)
#         ax2.annotate("Post-COVID spike", xy=(2022, ax2.get_ylim()[1] * 0.85),
#                      fontsize=9, ha="center", color=RED, fontstyle="italic")
#         ax2.set_xlabel("Year")
#         ax2.set_ylabel("Total Crimes")
#         ax2.legend(frameon=True)
#         ax2.set_xticks(sorted(temescal_yr["year"].unique()))
#         ax2.set_title("Full Trend: Temescal vs Control (2012-2023)", fontweight="bold")
#         plt.tight_layout()
#         st.pyplot(fig2)
#         plt.close()
#         st.markdown("The sharp divergence after 2020 is likely driven by pandemic-era "
#                     "disruptions and recovery effects rather than the 2015 upzoning alone.")
#
#     st.markdown("---")
#     st.markdown("**What factors explain this?** The crime regression shows which neighborhood "
#                 "characteristics drive crime rates:")
#
#     col1, col2 = st.columns(2)
#     keep = ["density", "log_median_value", "pct_residential", "log_income"]
#     labels = ["Housing Density", "Property Value (log)", "% Residential Land",
#               "Household Income (log)"]
#
#     with col1:
#         fig = coef_chart(m_total, keep, labels, "Total Crime Rate Drivers", BLUE)
#         st.pyplot(fig)
#         plt.close()
#     with col2:
#         fig = coef_chart(m_violent, keep, labels, "Violent Crime Rate Drivers", RED)
#         st.pyplot(fig)
#         plt.close()
