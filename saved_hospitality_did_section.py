# ============================================================
# Hospitality Task Force — DiD Regression Section
# Removed 2026-04-14 at user's request ("take out the DiD for hospitality for now")
# To re-insert: paste this block back into app.py tab3 (Hospitality Task Force)
#   after the 311 Complaints caption (line ~890) and before "Did Officials Acknowledge This?"
# Models needed: m_hosp_total, m_hosp_drug, m_hosp_dispatch, m_hosp_trend
#   (loaded via load_and_fit() at top of app.py)
# ============================================================

#     # ── DiD Regression ───
#     st.markdown("---")
#     st.markdown("#### Difference-in-Differences: Measuring the Displacement")
#     st.markdown("A DiD model compares the hospitality zone (treatment) to surrounding "
#                 "neighborhoods (control) before and after Feb 2025. The interaction term "
#                 "(`treated × post`) isolates the task force's causal effect.")
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         # Coefficient chart for the three DiD models
#         did_models = {
#             "Total Crime": m_hosp_total,
#             "Drug Offenses": m_hosp_drug,
#             "Dispatch Calls": m_hosp_dispatch,
#         }
#         labels_did = []
#         coefs_did = []
#         ci_lo = []
#         ci_hi = []
#         pvals_did = []
#         colors_did = []
#
#         for name, model in did_models.items():
#             c = model.params["treated_x_post"]
#             ci = model.conf_int().loc["treated_x_post"]
#             p = model.pvalues["treated_x_post"]
#             labels_did.append(name)
#             coefs_did.append(c)
#             ci_lo.append(ci[0])
#             ci_hi.append(ci[1])
#             pvals_did.append(p)
#             colors_did.append(GREEN if c < 0 else RED)
#
#         fig, ax = plt.subplots(figsize=(7, 3.5))
#         y_pos = range(len(labels_did))
#         errs = [[c - lo for c, lo in zip(coefs_did, ci_lo)],
#                 [hi - c for c, hi in zip(coefs_did, ci_hi)]]
#
#         ax.barh(y_pos, coefs_did, color=colors_did, alpha=0.85, height=0.5)
#         ax.errorbar(coefs_did, y_pos, xerr=errs, fmt="none", ecolor="black",
#                     capsize=4, linewidth=1.5)
#         ax.axvline(0, color="black", linewidth=1, linestyle="-")
#
#         for i, (coef, p) in enumerate(zip(coefs_did, pvals_did)):
#             stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
#             side = -15 if coef < 0 else 15
#             ax.annotate(f"{coef:+.0f}{stars}",
#                         xy=(coef, i), xytext=(side, 0),
#                         textcoords="offset points", va="center",
#                         fontsize=10, fontweight="bold",
#                         color="white" if abs(coef) > 50 else "black")
#
#         ax.set_yticks(list(y_pos))
#         ax.set_yticklabels(labels_did, fontsize=11)
#         ax.set_xlabel("DiD Estimate (treated × post)")
#         ax.set_title("Task Force Effect on Hospitality Zone\nvs Surrounding Neighborhoods",
#                     fontweight="bold")
#         plt.tight_layout()
#         st.pyplot(fig)
#         plt.close()
#
#         st.caption("Bars show the DiD coefficient with 95% CI. "
#                    "Negative = crime decreased in hospitality zone relative to controls. "
#                    "\\*** p < 0.001")
#
#     with col2:
#         st.markdown("**How to read this:**")
#         st.markdown(
#             "The DiD isolates the task force's effect by comparing "
#             "the hospitality zone to Mission + SoMa before and after Feb 2025."
#         )
#
#         for name, model in did_models.items():
#             coef = model.params["treated_x_post"]
#             p = model.pvalues["treated_x_post"]
#             post_ctrl = model.params["post"]
#             stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
#             st.markdown(f"**{name}:** `{coef:+.0f}/month`{stars}")
#
#         st.markdown("---")
#         st.markdown(
#             f"**The `post` coefficient tells the displacement story:** "
#             f"After Feb 2025, surrounding neighborhoods saw "
#             f"**+{m_hosp_total.params['post']:,.0f} total crimes/mo**, "
#             f"**+{m_hosp_drug.params['post']:,.0f} drug offenses/mo**, and "
#             f"**+{m_hosp_dispatch.params['post']:,.0f} dispatch calls/mo** "
#             f"-- while downtown dropped."
#         )
#
#         st.markdown("")
#         trend_coef = m_hosp_trend.params["treated_x_post"]
#         trend_p = m_hosp_trend.pvalues["treated_x_post"]
#         time_p = m_hosp_trend.pvalues["t"]
#         st.markdown(
#             f"**Robustness check:** With a time trend control, the task force "
#             f"effect holds at `{trend_coef:+.0f}/month` (p={trend_p:.4f}). "
#             f"The time trend itself is {'not ' if time_p > 0.05 else ''}significant "
#             f"(p={time_p:.2f}), confirming this is a policy effect, not a pre-existing trend."
#         )
