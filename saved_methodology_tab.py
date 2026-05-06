"""
============================================================
Methodology Tab — Saved for Refinement
============================================================

This file holds the full Methodology tab that was previously the third
tab of the dashboard. It was removed temporarily to give the user time
to refine the framing, data provenance descriptions, and limitations
language before re-introducing it to the live dashboard.

----- HOW TO RESTORE -----

When you're ready to put it back:

1. In app.py, find the st.tabs(...) call near the top of the page
   section. It currently reads:

       tab0, tab1, tab2 = st.tabs([
           "Overview",
           "Hospitality Task Force",
           "Office-to-Residential & DRD",
       ])

   Add "Methodology" back as a fourth tab and unpack it as `tab3`:

       tab0, tab1, tab2, tab3 = st.tabs([
           "Overview",
           "Hospitality Task Force",
           "Office-to-Residential & DRD",
           "Methodology",
       ])

2. Paste the `with tab3:` block from this file (everything below the
   "PASTE-IN BLOCK" marker, uncommented) at the bottom of app.py.

3. Re-add the line in the Overview tab's "Where to start" section that
   points readers to the Methodology tab. Currently the third column
   reads only DRD; add it back as a fourth column or replace the third.

----- WHY IT WAS REMOVED -----

The methodology page is structurally complete but the user wanted to
revisit:
- More precise data provenance language (especially clarifying which
  monthly figures came directly from SFPD vs. were assembled from
  journalism citations)
- Tightening the limitations section
- Possibly reorganizing the analytical-methods description

============================================================
PASTE-IN BLOCK (uncomment the `with tab3:` line and the body below
to restore as a Streamlit tab)
============================================================
"""

# with tab3:
#     st.subheader("Methodology")
#     st.markdown("This page documents the research question, analytical approach, "
#                 "data sources, and limitations underlying every chart in this dashboard.")
#
#     st.markdown("---")
#
#     # ══════════════════════════════════════════════
#     # RESEARCH QUESTION
#     # ══════════════════════════════════════════════
#
#     st.markdown("## Research Question")
#     st.info(
#         "San Francisco's *'Bring SF Back'* agenda promises broad recovery through "
#         "downtown revitalization. Examining its two flagship policies — the **Downtown "
#         "Revitalization District (DRD)** and the **Hospitality Task Force** — how are "
#         "the benefits and costs distributed across neighborhoods, and where do "
#         "measurable impacts diverge from public narratives about the city's recovery?"
#     )
#
#     col_a, col_b = st.columns(2)
#     with col_a:
#         st.markdown(
#             "**Why these two policies?**  \n"
#             "They are the Lurie administration's flagship downtown revitalization "
#             "interventions: one targets the *built environment* (housing supply via "
#             "office conversions), the other targets the *streetscape* (visible safety "
#             "via concentrated police presence). Both are framed as benefits to the "
#             "whole city. Both have measurable spillover effects on surrounding "
#             "neighborhoods that are rarely acknowledged in official communications."
#         )
#     with col_b:
#         st.markdown(
#             "**What this dashboard asks**  \n"
#             "1. What are the **measurable impacts** of each policy on housing and "
#             "public safety?  \n"
#             "2. **Who benefits** from these impacts, and **who bears the costs**?  \n"
#             "3. How do the **measurable impacts** compare to the **public narratives** "
#             "officials use to promote the policies?"
#         )
#
#     st.markdown("---")
#
#     # ══════════════════════════════════════════════
#     # ANALYTICAL APPROACH
#     # ══════════════════════════════════════════════
#
#     st.markdown("## Analytical Approach")
#     st.markdown("Each policy is analyzed using a parallel three-part structure:")
#
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.markdown(
#             "### Part 1: The Promise\n"
#             "What the policy *says* it will do, in officials' own words. "
#             "Sources: SF.gov press releases, State of the City addresses, "
#             "Mayor's social media, on-record statements."
#         )
#     with col2:
#         st.markdown(
#             "### Part 2: The Data\n"
#             "What the data actually shows. Monthly crime time series "
#             "(Hospitality Zone vs surrounding neighborhoods) for the Hospitality tab; "
#             "permit data, rent indices, and tax-increment math for the DRD tab."
#         )
#     with col3:
#         st.markdown(
#             "### Part 3: Who Benefits / Who Pays\n"
#             "Distributional analysis: who experiences the benefits "
#             "(visitors, voters, market-rate buyers) versus who absorbs the costs "
#             "(adjacent neighborhoods, lower-income residents, displaced populations)."
#         )
#
#     st.markdown("---")
#
#     # ══════════════════════════════════════════════
#     # ANALYTICAL METHODS
#     # ══════════════════════════════════════════════
#
#     st.markdown("## Analytical Methods")
#
#     st.markdown("### Hospitality Task Force")
#     st.markdown(
#         "We compare monthly crime, drug offense, and 911 dispatch volumes in the "
#         "Hospitality Zone against surrounding neighborhoods (Mission + SoMa) before "
#         "and after the Feb 2025 launch.  \n"
#         "- **Pre/post comparison** — averages and percent change in crime "
#         "categories before and after task-force deployment  \n"
#         "- **Cross-zone comparison** — parallel time series showing whether crime "
#         "decreased downtown but rose in surrounding zones  \n"
#         "- **Narrative coding** — 9 official statements coded for whether they "
#         "acknowledge displacement; 22 attributed resident quotes coded for "
#         "neighborhood, topic, and sentiment"
#     )
#     st.caption(
#         "**Supplementary analysis:** A Difference-in-Differences (DiD) regression "
#         "of the task force's effect is available as a standalone script in the "
#         "GitHub repo at `streamlit_app/saved_hospitality_did_section.py`. "
#         "Run it with `python saved_hospitality_did_section.py` for formal "
#         "confidence intervals, p-values, and a time-trend robustness check. "
#         "The DiD is omitted from the dashboard itself because the descriptive "
#         "time-series and metrics in the Hospitality tab already convey the "
#         "displacement story for a general audience."
#     )
#
#     st.markdown("### DRD")
#     st.markdown(
#         "No causal estimate is attempted for the DRD because the policy has "
#         "produced **zero completed conversions** as of the Financing Plan adoption "
#         "(Feb 2026). Instead, we analyze:  \n"
#         "- **Pipeline status** (filed / issued / completed) from SF Planning Dept "
#         "permit data  \n"
#         "- **Rent vs income gap** using Zumper rent data and Census ACS income data  \n"
#         "- **Tax-increment financial flows** ($1.22B district cap, 64.59% developer "
#         "share, 30-year max duration)"
#     )
#
#     st.markdown("---")
#
#     # ══════════════════════════════════════════════
#     # DATA SOURCES
#     # ══════════════════════════════════════════════
#
#     st.markdown("## Data Sources")
#
#     st.markdown("### Hospitality Task Force Tab")
#     st.markdown('''
#     | Dataset | Source | Coverage | Notes |
#     |---------|--------|----------|-------|
#     | Hospitality Zone monthly crime | SFPD Incident Reports | Jan 2024 – Mar 2026 | Pre/post task force comparison |
#     | Surrounding-neighborhood crime | SFPD Incident Reports | Jan 2024 – Mar 2026 | Mission + SoMa for cross-zone comparison |
#     | Drug offenses & dispatch calls | SFPD DMACC, GrowSF reporting | 2024 – 2026 | Drug offenses are NOT Part 1 crimes |
#     | 311 Complaints | DataSF 311 Cases | Monthly | 16th & Mission 10-year highs |
#     | Official statements | SF.gov press releases, KQED, Mission Local, ABC7, NBC, SF Standard | Feb 2025 – Jan 2026 | 9 statements coded for displacement mention |
#     | Resident sentiment | Mission Local, SF Standard, ABC7, NBC, Axios SF, El Tecolote | Feb 2025 – Apr 2026 | 22 attributed quotes |
#     | Demographics | Census ACS 5-Year | 2020 – 2024 | Hospitality Zone vs adjacent |
#     | City Survey (safety / police) | SF City Survey | 2023, n=2,500+ | By race and neighborhood |
#     | Public polling | GrowSF Pulse Poll | Feb 2025, Jul 2025 | Voter support for downtown patrols |
#     ''')
#
#     st.markdown("### DRD Tab")
#     st.markdown('''
#     | Dataset | Source | Coverage | Notes |
#     |---------|--------|----------|-------|
#     | Legislative history | AB 2488 (2024), AB 1445 (2025), SF Ord. 20-25, DRD Financing Plan (Feb 2026) | Statute + ordinance | Program rules |
#     | Boundaries | SF Office of Economic & Workforce Development | Feb 2026 | Downtown C-3 zoning districts |
#     | Development pipeline | SF Planning Dept building permits | Historical permits | Office-to-residential filings |
#     | Commercial / residential prices | CBRE, JLL, Cushman & Wakefield, Zumper, Zillow | Quarterly | Some interpolated |
#     | Vacancy data | SF.gov, JLL Office Market Reports | 2019 – 2025 | Office vacancy trend |
#     | Population | Census ACS 5-Year | 2019 – 2024 | COVID-era population loss |
#     ''')
#
#     st.markdown("### Methodology Tab (this page)")
#     st.markdown(
#         "All quotes are **direct attributions** from named sources, with publication "
#         "and date. No quotes are paraphrased or fabricated. Where speakers are "
#         'unnamed in source reporting (e.g., "a 30-year resident"), they are listed '
#         'as "Unnamed" with their identifying context preserved from the original.'
#     )
#
#     st.markdown("---")
#
#     # ══════════════════════════════════════════════
#     # KEY VARIABLES
#     # ══════════════════════════════════════════════
#
#     st.markdown("## Key Variables")
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         st.markdown("### Hospitality / Crime Variables")
#         st.markdown('''
#         | Variable | Definition |
#         |----------|-----------|
#         | **zone** | Hospitality Zone vs surrounding (Mission, SoMa) |
#         | **year_month** | Monthly time period |
#         | **total_crimes** | All reported crimes in the zone |
#         | **drug_offenses** | Drug-related offenses (NOT Part 1 — excluded from official SFPD dashboards) |
#         | **dispatch_calls** | 911 dispatch calls to the zone |
#         | **mentions_displacement** | Whether an official statement acknowledges displacement |
#         | **sentiment** | Negative / Mixed / Positive (manually coded from quote tone) |
#         ''')
#
#     with col2:
#         st.markdown("### DRD Variables")
#         st.markdown('''
#         | Parameter | Value |
#         |-----------|-------|
#         | **Tier 1 threshold** | First 1.5M sq ft (no local affordability requirements) |
#         | **Tier 2 threshold** | 1.5M – 7M sq ft (state minimums apply) |
#         | **Tier 3 threshold** | 7M+ sq ft (state + local requirements) |
#         | **Tax increment share** | ~64.59% of new property tax revenue returned to developer |
#         | **Max duration** | 30 years per project |
#         | **District-wide cap** | $1.22 billion total tax revenue diversion |
#         | **Enrollment deadline** | December 31, 2032 |
#         | **Min residential** | 60% of gross floor area must be residential |
#         | **Status** | 0 completed conversions as of Feb 2026 |
#         ''')
#
#     st.markdown("---")
#
#     # ══════════════════════════════════════════════
#     # LIMITATIONS
#     # ══════════════════════════════════════════════
#
#     st.markdown("## Limitations & Caveats")
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         st.markdown("### Data Limitations")
#         st.markdown(
#             "- **Crime data lag.** SFPD reports are subject to revision. The "
#             "underlying SFPD crime panel runs 2018 – 2023; monthly post-task-force "
#             "data (2024 – 2026) is drawn from supplementary SFPD and DMACC reporting.  \n"
#             "- **Drug offenses are not Part 1 crimes.** They are excluded from the "
#             "official SFPD dashboards officials cite. This is itself a finding, not "
#             "a flaw — but readers should note it.  \n"
#             "- **DRD is too new for outcome data.** With zero completed conversions, "
#             "the analysis necessarily relies on program design, pipeline activity, "
#             "and parallel cases — not realized outcomes.  \n"
#             "- **Resident quotes are non-random.** They come from journalism and "
#             "community meetings, which over-represent organized voices and "
#             "under-represent the silent majority on either side."
#         )
#
#     with col2:
#         st.markdown("### Causal Caveats")
#         st.markdown(
#             "- **Cross-zone comparison is descriptive, not causal.** The dashboard "
#             "shows crime trajectories diverging between the Hospitality Zone and "
#             "surrounding neighborhoods after Feb 2025. This is descriptive "
#             "evidence consistent with displacement; for a more formal causal "
#             "estimate, see the DiD script linked in *Analytical Methods*.  \n"
#             "- **Citywide crime trends overlap.** SF saw a citywide crime decline "
#             "through 2025. Some of the Hospitality Zone improvement reflects this "
#             "broader trend rather than the task force alone.  \n"
#             "- **Displacement is an interpretation.** The data show crime "
#             "increases in the Mission and SoMa coinciding with task-force "
#             "deployment downtown. The *causal pathway* (displacement) is "
#             "supported by SFPD's own statements (e.g., Cmdr. Lew, May 2025) "
#             "but cannot be proven by the time-series data alone.  \n"
#             "- **Rent and income data are point-in-time.** Snapshots of "
#             "affordability gaps don't capture how the DRD will play out over "
#             "its 30-year window."
#         )
#
#     st.markdown("---")
#
#     # ══════════════════════════════════════════════
#     # REPLICATION
#     # ══════════════════════════════════════════════
#
#     st.markdown("## Replication")
#     st.markdown(
#         "All data files are committed to the project's GitHub repository. "
#         "All dashboard code is in `app.py`. To reproduce: clone the repo, "
#         "install requirements, run `streamlit run app.py`. The supplementary "
#         "DiD analysis can be reproduced by running "
#         "`python saved_hospitality_did_section.py` from the same directory."
#     )
