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
NAVY = "#1E3A8A"
SLATE_900 = "#0F172A"
SLATE_700 = "#334155"
SLATE_500 = "#64748B"
SLATE_200 = "#E2E8F0"
SLATE_50 = "#F8FAFC"

# Consistent zone colors used across the dashboard
ZONE_COLORS = {
    "Hospitality Zone": "#16A34A",      # green = treated/downtown
    "Mission District": "#DC2626",      # red = absorbing displacement
    "SoMa (Southern)": "#F59E0B",       # amber = also absorbing
    "Rest of SF": "#2563EB",            # blue = baseline comparison
}


# ── GLOBAL CSS — modern typography, spacing, and surface design ──
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Instrument+Serif&display=swap" rel="stylesheet">

    <style>
    /* ── 1. Modern font stack — Inter for UI, Instrument Serif for headlines ── */
    html, body, [class*="css"], .stApp, .stMarkdown, .stMarkdown p,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    .stTabs, .stButton, .stSelectbox, .stTextInput {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                     Helvetica, Arial, sans-serif !important;
        font-feature-settings: 'cv11', 'ss01', 'ss03';
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* ── 2. App background — softer warm off-white ── */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #FAFAF9;
    }
    .main .block-container {
        padding-top: 2.6rem;
        max-width: 1180px;
    }

    /* ── 3. Refined typographic hierarchy ── */
    h1, h2, h3, h4 {
        color: #0A0A0A;
        letter-spacing: -0.022em;
        font-weight: 700;
    }
    h1 {
        font-size: 2.5rem;
        line-height: 1.05;
        font-weight: 800;
    }
    h2 {
        font-size: 1.7rem;
        line-height: 1.2;
        margin-top: 1.6em;
        margin-bottom: 0.5em;
        letter-spacing: -0.024em;
    }
    h3 {
        font-size: 1.35rem;
        line-height: 1.3;
        margin-top: 1.3em;
        font-weight: 700;
    }
    h4 {
        font-size: 1.0rem;
        font-weight: 600;
        color: #27272A;
        letter-spacing: -0.01em;
    }

    /* ── 4. Body text — readable, slightly larger ── */
    .stMarkdown, .stMarkdown p, p {
        font-size: 1.0rem;
        line-height: 1.62;
        color: #27272A;
        font-weight: 400;
    }

    /* ── 5. Captions — modern muted styling ── */
    [data-testid="stCaptionContainer"] {
        font-size: 0.875rem;
        color: #71717A;
        line-height: 1.55;
    }

    /* ── 6. Metric components — refined display ── */
    [data-testid="stMetricValue"] {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0A0A0A;
        letter-spacing: -0.02em;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        font-weight: 600;
        color: #71717A;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.92rem;
        font-weight: 500;
    }

    /* ── 7. Expander — softer modern styling ── */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        background-color: #F4F4F5;
        border-radius: 10px;
        font-weight: 600;
        color: #1E3A8A;
        border: 1px solid #E4E4E7;
        padding: 12px 18px !important;
        transition: background-color 0.15s ease;
    }
    [data-testid="stExpander"] summary:hover {
        background-color: #ECECEE;
    }
    [data-testid="stExpander"] {
        border: none;
        box-shadow: none;
    }

    /* ── 8. Blockquote — refined pull quote ── */
    blockquote {
        background: linear-gradient(135deg, #FAFAF9 0%, #F4F4F5 100%);
        border-left: 3px solid #7C3AED;
        padding: 16px 24px;
        border-radius: 4px;
        margin: 14px 0;
        font-size: 1.0rem;
        color: #27272A;
        font-style: normal;
    }

    /* ── 9. Tab navigation — sleeker ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E4E4E7;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500;
        font-size: 0.95rem;
        color: #71717A;
        padding-top: 12px;
        padding-bottom: 12px;
        letter-spacing: -0.005em;
        transition: color 0.15s ease;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #27272A;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #0A0A0A;
        font-weight: 600;
        border-bottom-color: #7C3AED !important;
        border-bottom-width: 2.5px !important;
    }

    /* ── 10. Native st.info / st.success / st.warning — modernize ── */
    [data-testid="stAlert"] {
        border-radius: 12px;
        border-left-width: 4px;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── 11. Buttons — modern outline/fill ── */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500;
        border-radius: 8px;
        border: 1px solid #E4E4E7;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #7C3AED;
        color: #7C3AED;
    }

    /* ── 12. Divider — softer ── */
    hr {
        border: 0;
        height: 1px;
        background-color: #E4E4E7;
        margin: 2em 0;
    }

    /* ── 13. Code blocks — modern surface ── */
    code {
        background-color: #F4F4F5;
        border: 1px solid #E4E4E7;
        border-radius: 4px;
        padding: 1px 6px;
        font-size: 0.88em;
        color: #18181B;
        font-family: 'JetBrains Mono', 'SF Mono', Menlo, monospace;
    }

    /* ── 14. Selectbox — cleaner ── */
    .stSelectbox label {
        font-weight: 500;
        font-size: 0.88rem;
        color: #52525B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Modern matplotlib defaults ────────────────
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": "#0A0A0A",
    "axes.labelcolor": "#52525B",
    "axes.labelsize": 10,
    "axes.edgecolor": "#E4E4E7",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#E4E4E7",
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "xtick.color": "#71717A",
    "ytick.color": "#71717A",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "lines.linewidth": 2.4,
    "lines.markersize": 5,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "figure.facecolor": "#FAFAF9",
    "axes.facecolor": "#FAFAF9",
    "savefig.facecolor": "#FAFAF9",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── HELPERS for visual consistency ────────────
def style_axes(ax, title=None, xlabel=None, ylabel=None, grid=True):
    """Apply consistent styling to a matplotlib axis."""
    if title:
        ax.set_title(title, fontweight="bold", fontsize=12, color=SLATE_900,
                     loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=SLATE_700)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=SLATE_700)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SLATE_200)
    ax.spines["bottom"].set_color(SLATE_200)
    ax.tick_params(colors=SLATE_700, labelsize=9)
    if grid:
        ax.grid(True, axis="y", linestyle="--", alpha=0.4, color=SLATE_200)
        ax.set_axisbelow(True)


def hero_stat_card(label, big_number, sub_text, accent_color):
    """Styled HTML card for a single dramatic statistic."""
    return f"""
    <div style="
        border-left: 5px solid {accent_color};
        background: linear-gradient(90deg, {accent_color}10 0%, white 80%);
        padding: 18px 22px;
        border-radius: 10px;
        height: 100%;
    ">
        <div style="font-size: 0.78em; color: {SLATE_500}; font-weight: 700;
                    letter-spacing: 1.2px; margin-bottom: 4px;">
            {label}
        </div>
        <div style="font-size: 2.1em; font-weight: 800; color: {accent_color};
                    line-height: 1.1; margin-bottom: 6px;">
            {big_number}
        </div>
        <div style="font-size: 0.92em; color: {SLATE_700}; line-height: 1.4;">
            {sub_text}
        </div>
    </div>
    """


def takeaway_box(title, body, color=PURPLE):
    """Color-accented callout card for a key takeaway."""
    return f"""
    <div style="
        background-color: {color}10;
        border-left: 5px solid {color};
        padding: 16px 22px;
        border-radius: 8px;
        margin: 14px 0 18px 0;
    ">
        <div style="font-size: 0.78em; color: {color}; font-weight: 700;
                    letter-spacing: 1.2px; margin-bottom: 6px;">
            {title}
        </div>
        <div style="font-size: 0.98em; color: {SLATE_900}; line-height: 1.5;">
            {body}
        </div>
    </div>
    """


def summary_card(eyebrow, headline_number, headline_label, body, accent_color):
    """
    Three-part executive-summary card combining a big stat with narrative framing.
    Used at the top of each policy tab to convey the full picture before details.
    """
    return f"""
    <div style="
        background: white;
        border-top: 5px solid {accent_color};
        border-radius: 10px;
        padding: 22px 24px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08),
                    0 1px 2px rgba(15, 23, 42, 0.04);
    ">
        <div style="font-size: 0.75em; color: {accent_color}; font-weight: 700;
                    letter-spacing: 1.5px; margin-bottom: 14px;">
            {eyebrow}
        </div>
        <div style="font-size: 2.2em; font-weight: 800; color: {SLATE_900};
                    line-height: 1.0; margin-bottom: 4px;">
            {headline_number}
        </div>
        <div style="font-size: 0.85em; color: {SLATE_500}; font-weight: 600;
                    margin-bottom: 12px; text-transform: uppercase;
                    letter-spacing: 0.5px;">
            {headline_label}
        </div>
        <div style="font-size: 0.95em; color: {SLATE_700}; line-height: 1.5;">
            {body}
        </div>
    </div>
    """



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
# Animated entrance: title fades in, then "?", then subtitle, then tabs.
st.markdown(
    """
    <style>
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.4); }
        to   { opacity: 1; transform: scale(1); }
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.05;
        margin: 0.4em 0 0 0;
        opacity: 0;
        animation: fadeUp 0.7s ease-out 0.1s forwards;
    }
    .hero-q {
        display: inline-block;
        color: #7C3AED;
        opacity: 0;
        animation: scaleIn 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) 1.0s forwards;
        transform-origin: bottom left;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #64748B;
        font-weight: 400;
        line-height: 1.45;
        margin: 0.5em 0 1.3em 0;
        max-width: 60ch;
        opacity: 0;
        animation: fadeUp 0.7s ease-out 1.8s forwards;
    }
    .hero-tabs-wrap {
        opacity: 0;
        animation: fadeUp 0.6s ease-out 2.6s forwards;
    }
    .hero-arrow-down {
        text-align: center;
        font-size: 1.5rem;
        color: #7C3AED;
        opacity: 0;
        animation: fadeUp 0.6s ease-out 2.6s forwards,
                   bobble 1.6s ease-in-out 3.4s infinite;
        margin: 0.4em 0 0.2em 0;
    }
    @keyframes bobble {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(6px); }
    }

    /* Optional: respect users who prefer reduced motion */
    @media (prefers-reduced-motion: reduce) {
        .hero-title, .hero-q, .hero-subtitle,
        .hero-tabs-wrap, .hero-arrow-down {
            animation: none;
            opacity: 1;
        }
    }
    </style>

    <div class="hero-title">
        Bringing SF Back<span class="hero-q">?</span>
    </div>
    <div class="hero-subtitle">
        Who benefits from downtown revitalization &mdash; and where does the
        data diverge from the narrative?
    </div>
    <div class="hero-arrow-down">&darr;</div>
    """,
    unsafe_allow_html=True,
)

# Tabs (wrapped so they fade in slightly after the subtitle)
st.markdown('<div class="hero-tabs-wrap">', unsafe_allow_html=True)
tab0, tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Hospitality Task Force",
    "Office-to-Residential & DRD",
    "Why is this important?",
])
st.markdown('</div>', unsafe_allow_html=True)


# Helper: render a styled "next-tab" call-to-action card at the bottom
# of each tab. Renders as a components.html iframe so the embedded JS can
# reach into the parent document, find the matching tab button by label,
# and click it — giving us programmatic tab switching that st.tabs doesn't
# expose natively.
import streamlit.components.v1 as _components

def next_tab_arrow(next_label, color, last=False):
    """Render a 'Next →' call-to-action that navigates to the next tab on click."""
    if last:
        body = (
            f'<div style="font-size: 0.78em; color: {SLATE_500}; '
            f'font-weight: 700; letter-spacing: 1.2px; margin-bottom: 4px;">'
            f"YOU'VE REACHED THE END</div>"
            f'<div style="font-size: 1.15em; font-weight: 700; '
            f'color: {SLATE_900};">Both policies, examined.</div>'
            f'<div style="font-size: 0.92em; color: {SLATE_700}; '
            f'margin-top: 6px;">Click any tab above to revisit a section.</div>'
        )
        clickable_attrs = ""
        cursor = "default"
    else:
        body = (
            f'<div style="font-size: 0.78em; color: {color}; '
            f'font-weight: 700; letter-spacing: 1.2px; margin-bottom: 4px;">'
            f'CONTINUE THE STORY</div>'
            f'<div style="font-size: 1.25em; font-weight: 700; '
            f'color: {SLATE_900};">'
            f'Next: {next_label} &nbsp;<span style="color: {color};">&rarr;</span>'
            f'</div>'
            f'<div style="font-size: 0.88em; color: {SLATE_500}; '
            f'margin-top: 6px;">Click here to jump to <b>{next_label}</b>.</div>'
        )
        clickable_attrs = f' data-target-tab="{next_label}" class="tab-link"'
        cursor = "pointer"

    html = f"""
    <div{clickable_attrs} style="
        margin-top: 16px;
        padding: 22px 26px;
        background: linear-gradient(135deg, {color}10 0%, white 100%);
        border-left: 5px solid {color};
        border-radius: 12px;
        cursor: {cursor};
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        font-family: 'Inter', -apple-system, sans-serif;
    "
    onmouseover="if (this.dataset.targetTab) {{ this.style.transform='translateX(4px)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.08)'; }}"
    onmouseout="this.style.transform='translateX(0)'; this.style.boxShadow='none';">
        {body}
    </div>
    <script>
    (function() {{
        const card = document.querySelector('.tab-link');
        if (!card) return;
        card.addEventListener('click', function() {{
            const label = card.dataset.targetTab;
            const parentDoc = window.parent.document;
            const tabs = parentDoc.querySelectorAll('button[role="tab"]');
            for (const t of tabs) {{
                if ((t.innerText || '').trim() === label) {{
                    t.click();
                    parentDoc.defaultView.scrollTo({{ top: 0, behavior: 'smooth' }});
                    break;
                }}
            }}
        }});
    }})();
    </script>
    """
    _components.html(html, height=160)


# ══════════════════════════════════════════════
# FLOATING "BACK TO TOP" BUTTON + AUTO-SCROLL ON TAB CHANGE
# (injected via component so JS actually runs)
# ══════════════════════════════════════════════
import streamlit.components.v1 as components

components.html(
    """
    <script>
    (function() {
        const parent = window.parent;
        const doc = parent.document;

        // Find every plausible scroll container Streamlit might use, plus
        // window/document. We'll scroll all of them on click — harmless to
        // scroll a non-scrolling element.
        const getScrollTargets = () => {
            const targets = [parent];
            const selectors = [
                'html', 'body',
                '[data-testid="stAppViewContainer"]',
                '[data-testid="stMain"]',
                'section.main',
                '.main',
                '.block-container',
                '[data-testid="stMainBlockContainer"]',
            ];
            selectors.forEach(sel => {
                const el = doc.querySelector(sel);
                if (el) targets.push(el);
            });
            return targets;
        };

        const scrollAllToTop = () => {
            getScrollTargets().forEach(t => {
                try {
                    if (t === parent || t === window) {
                        t.scrollTo({ top: 0, behavior: 'smooth' });
                    } else {
                        t.scrollTo({ top: 0, behavior: 'smooth' });
                        // fallback for elements that don't support smooth
                        t.scrollTop = 0;
                    }
                } catch (e) { /* ignore */ }
            });
        };

        // ── 1. Add the floating button (only once across reruns) ──
        if (!doc.getElementById('scroll-top-fab')) {
            const btn = doc.createElement('button');
            btn.id = 'scroll-top-fab';
            btn.innerHTML = '&uarr;';
            btn.title = 'Back to top';
            btn.style.cssText = `
                position: fixed;
                bottom: 28px;
                right: 28px;
                width: 50px;
                height: 50px;
                border-radius: 50%;
                background: linear-gradient(135deg, #7C3AED 0%, #1E3A8A 100%);
                color: white;
                border: none;
                cursor: pointer;
                box-shadow: 0 4px 14px rgba(124, 58, 237, 0.45);
                font-size: 1.4rem;
                font-weight: 700;
                z-index: 999999;
                transition: transform 0.18s ease, box-shadow 0.18s ease;
                opacity: 1;
                pointer-events: auto;
            `;
            btn.onmouseenter = () => {
                btn.style.transform = 'translateY(-3px)';
                btn.style.boxShadow = '0 6px 20px rgba(124, 58, 237, 0.6)';
            };
            btn.onmouseleave = () => {
                btn.style.transform = 'translateY(0)';
                btn.style.boxShadow = '0 4px 14px rgba(124, 58, 237, 0.45)';
            };
            btn.onclick = scrollAllToTop;
            doc.body.appendChild(btn);
        }

        // ── 2. Auto-scroll to top whenever a tab button is clicked ──
        const attachTabListeners = () => {
            const tabBtns = doc.querySelectorAll(
                '[data-baseweb="tab-list"] button[role="tab"]'
            );
            tabBtns.forEach(t => {
                if (!t._scrollTopAttached) {
                    t._scrollTopAttached = true;
                    t.addEventListener('click', () => {
                        setTimeout(scrollAllToTop, 80);
                        setTimeout(scrollAllToTop, 250);
                    });
                }
            });
        };
        attachTabListeners();

        // Re-attach if Streamlit re-renders
        const observer = new MutationObserver(attachTabListeners);
        observer.observe(doc.body, { childList: true, subtree: true });
    })();
    </script>
    """,
    height=0,
)


# ══════════════════════════════════════════════
# TAB 0: OVERVIEW
# ══════════════════════════════════════════════

with tab0:
    # ── Hero / Research Question ──
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1E3A8A 0%, #7C3AED 100%);
            padding: 36px 40px;
            border-radius: 12px;
            color: white;
            margin-top: 12px;
            margin-bottom: 28px;
        ">
            <div style="font-size: 0.95em; letter-spacing: 2px; opacity: 0.85; margin-bottom: 10px;">
                THE QUESTION
            </div>
            <div style="font-size: 1.55em; line-height: 1.5; font-weight: 500;">
                San Francisco's <i>"Bring SF Back"</i> agenda promises broad recovery
                through downtown revitalization. How are the
                <b>benefits and costs distributed</b> across neighborhoods,
                and where do <b>measurable impacts diverge from public narratives</b>
                about the city's recovery?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Two Policy Cards ──
    st.markdown("### Two flagship policies, examined closely")
    st.markdown(
        "Mayor Lurie's downtown revitalization agenda touches every part of city "
        "life. We focus on the two policies with the largest measurable impacts "
        "on residents -- one targeting **housing**, the other **public safety**."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style="
                border: 2px solid #DC2626;
                border-radius: 12px;
                padding: 24px;
                height: 100%;
                background-color: #FEF8F8;
            ">
                <div style="font-size: 0.85em; color: #DC2626; font-weight: 700; letter-spacing: 1.5px;">
                    POLICY 1 &nbsp;·&nbsp; PUBLIC SAFETY
                </div>
                <div style="font-size: 1.4em; font-weight: 700; margin-top: 6px; color: #1E293B;">
                    Hospitality Task Force
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="
                border: 2px solid #2563EB;
                border-radius: 12px;
                padding: 24px;
                height: 100%;
                background-color: #F8FAFC;
            ">
                <div style="font-size: 0.85em; color: #2563EB; font-weight: 700; letter-spacing: 1.5px;">
                    POLICY 2 &nbsp;·&nbsp; HOUSING
                </div>
                <div style="font-size: 1.4em; font-weight: 700; margin-top: 6px; color: #1E293B;">
                    Downtown Revitalization District
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("---")

    # ── How to Read This Dashboard ──
    st.markdown("### How each policy is examined")
    st.markdown(
        "Both policies are examined through the same lens: what officials "
        "say, what the data shows, and who benefits versus who pays."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style="
                background: #EFF6FF;
                border-left: 5px solid #2563EB;
                padding: 18px 22px;
                border-radius: 6px;
                height: 100%;
            ">
                <div style="font-size: 1.2em; font-weight: 700; color: #1E293B;">
                    The Promise
                </div>
                <div style="font-size: 0.93em; color: #475569; margin-top: 8px; line-height: 1.5;">
                    What officials say each policy will do &mdash; in mayoral
                    speeches, press releases, and on-record statements.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="
                background: #FFFBEB;
                border-left: 5px solid #F59E0B;
                padding: 18px 22px;
                border-radius: 6px;
                height: 100%;
            ">
                <div style="font-size: 1.2em; font-weight: 700; color: #1E293B;">
                    The Data
                </div>
                <div style="font-size: 0.93em; color: #475569; margin-top: 8px; line-height: 1.5;">
                    What the policy actually delivers &mdash; monthly crime
                    trends for Hospitality, and the subsidy mechanics and
                    affordability tiers behind the DRD.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div style="
                background: #F0FDF4;
                border-left: 5px solid #16A34A;
                padding: 18px 22px;
                border-radius: 6px;
                height: 100%;
            ">
                <div style="font-size: 1.2em; font-weight: 700; color: #1E293B;">
                    Who Benefits, Who Pays
                </div>
                <div style="font-size: 0.93em; color: #475569; margin-top: 8px; line-height: 1.5;">
                    Who each policy actually serves &mdash; and which residents
                    absorb the costs the official narratives leave out.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    next_tab_arrow("Hospitality Task Force", RED)


# ══════════════════════════════════════════════
# TAB 2: OFFICE-TO-RESIDENTIAL & DRD
# ══════════════════════════════════════════════

with tab2:
    st.subheader("Downtown Revitalization District (DRD)")
    st.markdown(
        "The Downtown Revitalization District (DRD) is a 30-year tax subsidy "
        "meant to convert empty downtown offices into housing. Developers keep "
        "up to **64.59%** of a converted building's new property taxes; the "
        "city forgoes that revenue, up to a **$1.22 billion** cap."
    )

    # ── Executive summary: the full picture in 30 seconds ──
    st.markdown(
        '<div style="font-size: 0.78em; color: ' + SLATE_500 + '; '
        'font-weight: 700; letter-spacing: 1.5px; margin: 8px 0 12px 0;">'
        'THE FULL PICTURE &mdash; SCROLL FOR DETAILS'
        '</div>',
        unsafe_allow_html=True,
    )

    hcol1, hcol2, hcol3 = st.columns(3)
    with hcol1:
        st.markdown(summary_card(
            "THE PROMISE",
            "$1.22B",
            "Promised investment in downtown revitalization",
            "Mayor Lurie's <i>Heart of the City</i> plan promises to "
            "transform empty downtown towers into thousands of new homes "
            "&mdash; restoring foot traffic, the tax base, and street life "
            "by returning up to <b>64.59%</b> of new property taxes to "
            "developers who convert offices into housing.",
            BLUE,
        ), unsafe_allow_html=True)
    with hcol2:
        st.markdown(summary_card(
            "THE DATA",
            "$3,665",
            "Median market rent in the first conversions",
            "The DRD's affordability is <b>phased</b>: the first "
            "<b>1,875 units</b> face <b>no affordability requirements</b>, "
            "and only later tiers begin to require modest affordable "
            "shares. \"Market-rate\" here means roughly <b>$3,665/month</b> "
            "&mdash; in line with what downtown apartments already rent "
            "for today.",
            RED,
        ), unsafe_allow_html=True)
    with hcol3:
        st.markdown(summary_card(
            "WHO BENEFITS / WHO PAYS",
            "0",
            "Affordable units in the first phase",
            "SF's housing crisis is about <b>affordability</b> &mdash; "
            "and this policy doesn't address it. <b>Developers</b> get "
            "tax savings, <b>high earners</b> get new units, and the "
            "residents actually priced out of downtown see no relief.",
            PURPLE,
        ), unsafe_allow_html=True)

    st.markdown("")
    st.markdown("---")

    # ── Timeline + Map (side-by-side, collapsible) ──
    ctx_left, ctx_right = st.columns(2)

    with ctx_left:
        with st.expander("&#128221;&nbsp; **How We Got Here** &mdash; legislative timeline", expanded=False):
            st.markdown(
                "| Year | Action |\n"
                "|------|--------|\n"
                "| **2024** | **AB 2488** — DRD framework established in state law |\n"
                "| **Mar 2025** | **Ordinance 20-25** — waives SF's inclusionary "
                "housing & impact fees for C-3 conversions |\n"
                "| **2025** | **AB 1445** — amends AB 2488 |\n"
                "| **Feb 2026** | **Financing Plan adopted** — defines boundaries, "
                "$1.22B district cap |"
            )
            st.caption(
                "Created under the Breed administration; the Lurie administration "
                "inherited the framework and made key implementation choices — "
                "including waiving local affordable housing requirements."
            )

    with ctx_right:
        with st.expander("&#128205;&nbsp; **District Boundaries** &mdash; where the DRD applies", expanded=False):
            st.image("data/drd_boundary_map.png", use_container_width=True)
            st.caption(
                "Approximate boundary based on AB 2488 §62450(h) and the Financing "
                "Plan (Feb 2026). Covers downtown C-3 zoning districts."
            )

    st.markdown("---")

    # ══════════════════════════════════════════════════════
    # PART 1: THE PROMISE
    # ══════════════════════════════════════════════════════
    st.markdown(
        f'<div style="display:inline-block; padding: 4px 12px; '
        f'background-color: {BLUE}15; color: {BLUE}; '
        f'border-radius: 20px; font-size: 0.8em; font-weight: 700; '
        f'letter-spacing: 1px; margin-bottom: 6px;">THE PROMISE</div>',
        unsafe_allow_html=True,
    )
    st.markdown("## The Stated Goal")
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
    # PART 2: THE DATA
    # ══════════════════════════════════════════════════════
    st.markdown(
        f'<div style="display:inline-block; padding: 4px 12px; '
        f'background-color: {RED}15; color: {RED}; '
        f'border-radius: 20px; font-size: 0.8em; font-weight: 700; '
        f'letter-spacing: 1px; margin-bottom: 6px;">THE DATA</div>',
        unsafe_allow_html=True,
    )
    st.markdown("## Policy Structure")
    st.markdown(
        "The DRD has two structural pieces: **how the money flows** between "
        "the city and developers, and **what affordability is required** at "
        "each tier."
    )

    fin_col, aff_col = st.columns(2, gap="large")

    # ─── FINANCING ──────────────────────────────
    with fin_col:
        st.markdown("### Financing")
        st.markdown(
            "When a developer converts an office to housing, the property's "
            "value &mdash; and its property taxes &mdash; go up. The DRD "
            "<b>returns up to 64.59%</b> of that increase to the developer "
            "for up to <b>30 years</b>.",
            unsafe_allow_html=True,
        )

        # Visual: split bar showing 64.59 / 35.41 distribution
        st.markdown(
            f"""
            <div style="margin: 22px 0 14px 0;">
                <div style="font-size: 0.78em; color: {SLATE_500};
                            font-weight: 700; letter-spacing: 1px;
                            margin-bottom: 8px;">
                    HOW EACH NEW TAX DOLLAR IS SPLIT
                </div>
                <div style="display: flex; height: 64px; border-radius: 10px;
                            overflow: hidden;
                            box-shadow: 0 1px 3px rgba(15,23,42,0.10);">
                    <div style="background: {BLUE}; color: white;
                                flex: 64.59;
                                display: flex; flex-direction: column;
                                justify-content: center; align-items: center;
                                padding: 0 10px;">
                        <div style="font-size: 1.35em; font-weight: 800;
                                    line-height: 1;">64.59%</div>
                        <div style="font-size: 0.78em; font-weight: 600;
                                    opacity: 0.95; margin-top: 3px;">
                            to Developer
                        </div>
                    </div>
                    <div style="background: {GREEN}; color: white;
                                flex: 35.41;
                                display: flex; flex-direction: column;
                                justify-content: center; align-items: center;
                                padding: 0 10px;">
                        <div style="font-size: 1.35em; font-weight: 800;
                                    line-height: 1;">35.41%</div>
                        <div style="font-size: 0.78em; font-weight: 600;
                                    opacity: 0.95; margin-top: 3px;">
                            to City
                        </div>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between;
                            margin-top: 8px; font-size: 0.78em;
                            color: {SLATE_500};">
                    <span>30-year subsidy</span>
                    <span>Schools, transit, services</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div style='font-size: 0.92em; color: {SLATE_700}; "
            f"margin-top: 6px;'><b>District cap:</b> $1.22 billion in "
            f"total tax revenue over 30 years.</div>",
            unsafe_allow_html=True,
        )

    # ─── AFFORDABILITY ──────────────────────────
    with aff_col:
        st.markdown("### Affordability")
        st.markdown(
            "Affordability requirements rise in tiers as enrollment grows. "
            "The first projects face none &mdash; later ones face more, "
            "but only if anyone gets that far.",
            unsafe_allow_html=True,
        )

        # Tier ladder — three stacked cards, color-coded for severity
        tiers = [
            {
                "n": "1",
                "color": GREEN,
                "threshold": "First 1.5M sq ft (~1,875 units)",
                "required_pct": "0%",
                "required_text": "<b>No affordability required</b> &mdash; "
                                 "100% market-rate allowed",
            },
            {
                "n": "2",
                "color": AMBER,
                "threshold": "1.5M &ndash; 7M sq ft",
                "required_pct": "5&ndash;10%",
                "required_text": "State minimums only "
                                 "(local SF inclusionary still waived)",
            },
            {
                "n": "3",
                "color": RED,
                "threshold": "7M+ sq ft",
                "required_pct": "~25%",
                "required_text": "State minimums <b>plus</b> full SF "
                                 "inclusionary requirements",
            },
        ]

        cards_html = '<div style="margin-top: 14px;">'
        for t in tiers:
            cards_html += (
                f'<div style="display:flex;align-items:stretch;margin-bottom:10px;'
                f'border-radius:10px;overflow:hidden;'
                f'box-shadow:0 1px 3px rgba(15,23,42,0.08),0 1px 2px rgba(15,23,42,0.04);'
                f'background:white;">'
                f'<div style="background:{t["color"]};color:white;padding:14px 12px;'
                f'min-width:78px;display:flex;flex-direction:column;'
                f'justify-content:center;align-items:center;">'
                f'<div style="font-size:0.65em;letter-spacing:1.2px;'
                f'font-weight:700;opacity:0.9;">TIER</div>'
                f'<div style="font-size:1.9em;font-weight:800;line-height:1;">{t["n"]}</div>'
                f'</div>'
                f'<div style="flex:1;padding:12px 16px;display:flex;'
                f'flex-direction:column;justify-content:center;">'
                f'<div style="font-size:0.78em;color:{SLATE_500};'
                f'font-weight:600;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin-bottom:4px;">{t["threshold"]}</div>'
                f'<div style="font-size:0.95em;color:{SLATE_900};line-height:1.4;">'
                f'<span style="color:{t["color"]};font-weight:800;">'
                f'{t["required_pct"]} affordable</span>'
                f'&nbsp;&middot;&nbsp; {t["required_text"]}'
                f'</div>'
                f'</div>'
                f'</div>'
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

    # ─── Synthesis: The Incentive Race (three scenarios) ──
    st.markdown(
        f"<div style='font-size: 0.78em; color: {PURPLE}; font-weight: 700; "
        f"letter-spacing: 1.5px; margin: 30px 0 6px 0;'>"
        f"THE INCENTIVE RACE</div>"
        f"<div style='font-size: 1.18em; font-weight: 700; "
        f"color: {SLATE_900}; margin-bottom: 8px;'>"
        f"Why early, market-rate conversions win &mdash; and late, affordable "
        f"ones get squeezed out.</div>"
        f"<div style='font-size: 0.95em; color: {SLATE_700}; line-height: 1.55; "
        f"margin-bottom: 18px;'>"
        f"The $1.22B subsidy pool drains as conversions enroll, and "
        f"affordability requirements rise at each tier. Compare what a "
        f"developer faces at three points in the program:</div>",
        unsafe_allow_html=True,
    )

    scenarios = [
        {
            "stage": "EARLY",
            "tier": "Tier 1",
            "color": GREEN,
            "pool": "100%",
            "pool_sub": "$1.22B available",
            "required": "0%",
            "required_sub": "No affordability obligation",
            "verdict": "HIGH INCENTIVE",
            "blurb": "Maximum subsidy with zero affordability cost &mdash; "
                     "the most attractive scenario.",
        },
        {
            "stage": "MIDDLE",
            "tier": "Tier 2",
            "color": AMBER,
            "pool": "~50%",
            "pool_sub": "Pool partly depleted",
            "required": "5&ndash;10%",
            "required_sub": "State minimums kick in",
            "verdict": "MODERATE",
            "blurb": "Smaller subsidy paired with a new affordability cost "
                     "&mdash; incentive shrinks.",
        },
        {
            "stage": "LATE",
            "tier": "Tier 3",
            "color": RED,
            "pool": "0%",
            "pool_sub": "Pool exhausted",
            "required": "~25%",
            "required_sub": "Full SF inclusionary",
            "verdict": "MINIMAL",
            "blurb": "No subsidy left and the highest affordability bar "
                     "&mdash; least viable for developers.",
        },
    ]

    sc_cols = st.columns(3, gap="medium")
    for col, s in zip(sc_cols, scenarios):
        with col:
            html = (
                f'<div style="background:white;border-top:5px solid {s["color"]};'
                f'border-radius:10px;padding:18px 20px;height:100%;'
                f'box-shadow:0 1px 3px rgba(15,23,42,0.08),0 1px 2px rgba(15,23,42,0.04);">'
                f'<div style="font-size:0.72em;color:{s["color"]};font-weight:800;'
                f'letter-spacing:1.5px;">{s["stage"]}</div>'
                f'<div style="font-size:0.92em;color:{SLATE_500};font-weight:600;'
                f'margin-bottom:18px;">{s["tier"]}</div>'
                f'<div style="font-size:0.7em;color:{SLATE_500};font-weight:700;'
                f'letter-spacing:1px;text-transform:uppercase;">Subsidy pool</div>'
                f'<div style="font-size:1.6em;font-weight:800;color:{BLUE};line-height:1.1;">'
                f'{s["pool"]}</div>'
                f'<div style="font-size:0.78em;color:{SLATE_500};margin-bottom:14px;">'
                f'{s["pool_sub"]}</div>'
                f'<div style="font-size:0.7em;color:{SLATE_500};font-weight:700;'
                f'letter-spacing:1px;text-transform:uppercase;">Affordability required</div>'
                f'<div style="font-size:1.6em;font-weight:800;color:{PURPLE};line-height:1.1;">'
                f'{s["required"]}</div>'
                f'<div style="font-size:0.78em;color:{SLATE_500};margin-bottom:16px;">'
                f'{s["required_sub"]}</div>'
                f'<div style="border-top:1px solid {SLATE_200};margin-bottom:14px;"></div>'
                f'<div style="display:inline-block;padding:5px 14px;'
                f'background:{s["color"]};color:white;border-radius:6px;'
                f'font-weight:800;letter-spacing:1px;font-size:0.82em;'
                f'margin-bottom:10px;">{s["verdict"]}</div>'
                f'<div style="font-size:0.88em;color:{SLATE_700};line-height:1.4;">'
                f'{s["blurb"]}</div>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

    st.caption(
        "Illustrative &mdash; pool depletion is modeled linearly to the "
        "Tier 3 threshold; the actual rate depends on per-project tax yields, "
        "which vary by building. The visual captures the direction of the "
        "dynamic, not the precise magnitude."
    )

    # ══════════════════════════════════════════════════════
    # PART 3: WHO BENEFITS, WHO PAYS
    # ══════════════════════════════════════════════════════

    st.markdown("---")
    st.markdown(
        f'<div style="display:inline-block; padding: 4px 12px; '
        f'background-color: {PURPLE}15; color: {PURPLE}; '
        f'border-radius: 20px; font-size: 0.8em; font-weight: 700; '
        f'letter-spacing: 1px; margin-bottom: 6px;">WHO BENEFITS, WHO PAYS</div>',
        unsafe_allow_html=True,
    )
    st.markdown("## The Affordability Gap")
    st.markdown(
        "**Who is this housing actually for** &mdash; and what does the "
        "city give up to deliver it?"
    )

    # Side-by-side comparison: Required income | Gap | Median income
    st.markdown(
        f'<div style="display:flex;align-items:stretch;gap:0;margin-top:18px;'
        f'border-radius:12px;overflow:hidden;'
        f'box-shadow:0 1px 3px rgba(15,23,42,0.10),0 1px 2px rgba(15,23,42,0.05);">'
        f'<div style="flex:1;background:white;border-top:5px solid {PURPLE};'
        f'padding:22px 26px;">'
        f'<div style="font-size:0.72em;color:{PURPLE};font-weight:800;'
        f'letter-spacing:1.5px;">REQUIRED INCOME</div>'
        f'<div style="font-size:2.6em;font-weight:800;color:{SLATE_900};'
        f'line-height:1.0;margin:8px 0 6px 0;">$146,600</div>'
        f'<div style="font-size:0.85em;color:{SLATE_500};font-weight:600;'
        f'line-height:1.4;">Per year, to afford $3,665/mo rent at the '
        f'standard 30%-of-income rule</div></div>'
        f'<div style="flex:0 0 auto;background:{SLATE_50};display:flex;'
        f'flex-direction:column;justify-content:center;align-items:center;'
        f'padding:18px 26px;border-left:1px solid {SLATE_200};'
        f'border-right:1px solid {SLATE_200};">'
        f'<div style="font-size:0.7em;color:{RED};font-weight:800;'
        f'letter-spacing:1.5px;">GAP</div>'
        f'<div style="font-size:1.5em;font-weight:800;color:{RED};'
        f'line-height:1.1;margin-top:4px;">&minus;$6,799</div>'
        f'<div style="font-size:0.72em;color:{SLATE_500};font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.6px;margin-top:2px;">'
        f'/yr short</div></div>'
        f'<div style="flex:1;background:white;border-top:5px solid {RED};'
        f'padding:22px 26px;">'
        f'<div style="font-size:0.72em;color:{RED};font-weight:800;'
        f'letter-spacing:1.5px;">SF MEDIAN HOUSEHOLD</div>'
        f'<div style="font-size:2.6em;font-weight:800;color:{SLATE_900};'
        f'line-height:1.0;margin:8px 0 6px 0;">$139,801</div>'
        f'<div style="font-size:0.85em;color:{SLATE_500};font-weight:600;'
        f'line-height:1.4;">Annual income, US Census ACS '
        f'(2019&ndash;2023 5-year estimates)</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Punchline callout
    st.markdown(
        f'<div style="margin-top:18px;padding:18px 22px;'
        f'background:linear-gradient(135deg,{PURPLE}10 0%,white 100%);'
        f'border-left:5px solid {PURPLE};border-radius:8px;">'
        f'<div style="font-size:0.78em;color:{PURPLE};font-weight:700;'
        f'letter-spacing:1.5px;margin-bottom:6px;">PUNCHLINE</div>'
        f'<div style="font-size:1.05em;color:{SLATE_900};line-height:1.55;">'
        f'<b>SF&apos;s median household can&apos;t afford the median DRD '
        f'unit.</b> The first 1,875 units &mdash; with no affordability '
        f'requirements &mdash; will serve households earning above the '
        f'city&apos;s middle. The renters priced out today are not the '
        f'ones this policy is built for.</div></div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Sources: $3,665 SF median rent (Zumper, 2024&ndash;25). "
        "$139,801 SF median household income (US Census ACS 2019&ndash;2023, "
        "5-year estimates). $146,600 derived from rent at the standard "
        "30%-of-income housing affordability rule."
    )

    next_tab_arrow("Why is this important?", SLATE_500)


# ══════════════════════════════════════════════
# TAB 1: HOSPITALITY TASK FORCE
# ══════════════════════════════════════════════

with tab1:
    st.subheader("Downtown Hospitality Safety Task Force (Feb 2025)")
    st.markdown(
        "A police unit created to make downtown feel safer for visitors and "
        "conventions. Officers patrol Union Square, Moscone, and Yerba Buena "
        "Gardens **20 hours a day, 365 days a year**."
    )

    # Pre-compute key numbers for hero stats
    hosp_zone = hosp_monthly[hosp_monthly["zone"] == "Hospitality Zone"].copy()
    rest_sf = hosp_monthly[hosp_monthly["zone"] == "Rest of SF"].copy()
    months = hosp_zone["year_month"].values
    feb_idx = list(months).index("2025-02") if "2025-02" in months else None
    pre = hosp_zone.iloc[:feb_idx]
    post = hosp_zone.iloc[feb_idx:]
    pre_avg = pre["total_crimes"].mean()
    post_avg = post["total_crimes"].mean()
    pct_change = ((post_avg - pre_avg) / pre_avg) * 100

    # ── Executive summary: the full picture in 30 seconds ──
    st.markdown(
        '<div style="font-size: 0.78em; color: ' + SLATE_500 + '; '
        'font-weight: 700; letter-spacing: 1.5px; margin: 8px 0 12px 0;">'
        'THE FULL PICTURE &mdash; SCROLL FOR DETAILS'
        '</div>',
        unsafe_allow_html=True,
    )

    hcol1, hcol2, hcol3 = st.columns(3)
    with hcol1:
        st.markdown(summary_card(
            "THE PROMISE",
            f"{pct_change:+.0f}%",
            "Crime in the Hospitality Zone",
            "Officials launched the task force in <b>Feb 2025</b> and "
            "made the downtown crime drop the centerpiece of SF's recovery "
            "narrative. Crime really did fall.",
            GREEN,
        ), unsafe_allow_html=True)
    with hcol2:
        st.markdown(summary_card(
            "THE DATA",
            "Part 1 only",
            "What &quot;crime is down&quot; actually measures",
            "&quot;Crime is down&quot; refers only to <b>Part 1 crimes</b> "
            "&mdash; homicide, robbery, larceny. It doesn't include drug "
            "offenses, disorderly conduct, or 911 calls &mdash; the "
            "categories that surged in the Mission and SoMa after Feb 2025. "
            "The crime didn't disappear; it just <b>hid in the categories "
            "and neighborhoods the official narrative chose to overlook</b>.",
            RED,
        ), unsafe_allow_html=True)
    with hcol3:
        st.markdown(summary_card(
            "WHO BENEFITS / WHO PAYS",
            "2 of 9",
            "Statements acknowledge displacement",
            "Visitors and voters see a safer downtown. Lower-income "
            "residents in adjacent neighborhoods absorb the cost &mdash; "
            "a gap officials rarely talk about publicly.",
            PURPLE,
        ), unsafe_allow_html=True)

    st.markdown("")
    st.markdown("---")

    # ══════════════════════════════════════════════
    # PART 1: THE PROMISE
    # ══════════════════════════════════════════════
    st.markdown(
        f'<div style="display:inline-block; padding: 4px 12px; '
        f'background-color: {GREEN}15; color: {GREEN}; '
        f'border-radius: 20px; font-size: 0.8em; font-weight: 700; '
        f'letter-spacing: 1px; margin-bottom: 6px;">THE PROMISE</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Crime Goes Down in the Hospitality Zone")
    st.markdown(
        '<div style="color: ' + SLATE_500 + '; font-size: 1.0em; '
        'margin-bottom: 14px;">'
        "The task force's success, as officials measure it."
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.7, 1])

    with col1:
        # Part 1 series (Property + Violent) — what officials cite
        part1_series = (hosp_zone["Property"].values
                        + hosp_zone["Violent"].values)

        part1_pre_avg = part1_series[:feb_idx].mean()
        part1_post_avg = part1_series[feb_idx:].mean()
        part1_change = ((part1_post_avg - part1_pre_avg) / part1_pre_avg) * 100

        fig, ax = plt.subplots(figsize=(8, 4.5))

        # Shaded pre/post regions
        if feb_idx is not None:
            ax.axvspan(-0.5, feb_idx, color=SLATE_200, alpha=0.25, zorder=0)
            ax.axvspan(feb_idx, len(months) - 0.5, color=GREEN,
                       alpha=0.07, zorder=0)

        # Part 1 only — the categories officials cite
        ax.plot(range(len(hosp_zone)), part1_series,
                color=RED, marker="o", linewidth=2.8, markersize=6,
                zorder=4)

        # Vertical Feb 2025 marker
        if feb_idx is not None:
            ax.axvline(feb_idx, color=SLATE_700, linewidth=1.5,
                       linestyle="--", alpha=0.6)
            ax.text(feb_idx + 0.15, ax.get_ylim()[1] * 0.96,
                    "Task Force\nlaunches",
                    fontsize=9, ha="left", va="top",
                    color=SLATE_700, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35",
                              facecolor="white", edgecolor="none",
                              alpha=0.95))

        ax.set_xticks(range(0, len(months), 3))
        ax.set_xticklabels([months[i][2:] for i in range(0, len(months), 3)],
                           rotation=0, ha="center", fontsize=9)
        style_axes(ax, ylabel="Part 1 crimes in the Hospitality Zone / month")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        if feb_idx is not None:
            st.markdown(
                '<div style="font-size: 0.78em; color: ' + SLATE_500 + '; '
                'font-weight: 700; letter-spacing: 1.2px; '
                'margin: 6px 0 12px 0;">'
                'CHANGE SINCE TASK FORCE'
                '</div>',
                unsafe_allow_html=True,
            )

            pre_part1 = (pre["Property"] + pre["Violent"]).mean()
            post_part1 = (post["Property"] + post["Violent"]).mean()
            part1_change_metric = ((post_part1 - pre_part1) / pre_part1) * 100

            accent = GREEN if part1_change_metric < 0 else RED
            st.markdown(
                f"""
                <div style="
                    margin-bottom: 10px;
                    padding: 22px 22px;
                    background: white;
                    border-left: 4px solid {accent};
                    border-radius: 6px;
                    box-shadow: 0 1px 2px rgba(15,23,42,0.05);
                ">
                    <div style="font-size: 0.78em; color: {SLATE_500};
                                font-weight: 600; letter-spacing: 0.5px;
                                text-transform: uppercase; margin-bottom: 6px;">
                        Part 1 Crime
                    </div>
                    <div style="font-size: 2.8em; font-weight: 800;
                                color: {accent}; line-height: 1.0;">
                        {part1_change_metric:+.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "This is exactly what officials reference when they say "
                "crime is down. **Part 1 (Index) crimes** include: "
                "homicide, rape, robbery, aggravated assault, burglary, "
                "larceny-theft, motor vehicle theft, and arson."
            )

    # ── Officials celebrate (now in expander) ───
    statements = pd.read_csv("data/official_statements.csv")
    celebrate = statements[statements["mentions_displacement"] == "No"]

    with st.expander(
        f"&#128172;&nbsp; Read what officials said &mdash; "
        f"{len(celebrate)} public statements celebrating the policy",
        expanded=False,
    ):
        for _, row in celebrate.iterrows():
            source_link = (f"[{row['source']}]({row['url']})"
                           if row.get('url') and pd.notna(row.get('url'))
                           else f"*{row['source']}*")
            st.markdown(
                f"> \"{row['quote']}\"  \n"
                f"> -- **{row['speaker']}**, {row['event']} ({row['date']}) "
                f"| {source_link}"
            )

    st.markdown(
        takeaway_box(
            "TAKEAWAY",
            f"Crime in the Hospitality Zone fell <b>{pct_change:+.0f}%</b> "
            f"after the task force launched. Officials made it a centerpiece "
            f"narrative. <b>If you stop here, the policy looks like a clear "
            f"success.</b>",
            color=GREEN,
        ),
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ══════════════════════════════════════════════
    # PART 2: BUT THAT'S NOT THE WHOLE STORY
    # ══════════════════════════════════════════════

    st.markdown(
        f'<div style="display:inline-block; padding: 4px 12px; '
        f'background-color: {RED}15; color: {RED}; '
        f'border-radius: 20px; font-size: 0.8em; font-weight: 700; '
        f'letter-spacing: 1px; margin-bottom: 6px;">THE FULL STORY</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Crime Didn't Disappear &mdash; It Moved")
    st.markdown(
        '<div style="color: ' + SLATE_500 + '; font-size: 1.0em; '
        'margin-bottom: 14px;">'
        "The neighborhoods surrounding the Hospitality Zone absorbed what "
        "downtown pushed out."
        '</div>',
        unsafe_allow_html=True,
    )

    disp = pd.read_csv("data/displacement_crime.csv")

    months_d = disp[disp["zone"] == "Hospitality Zone"]["year_month"].values
    feb_idx_d = list(months_d).index("2025-02") if "2025-02" in months_d else None

    # ── Toggle: Mission | SoMa ──
    zone_choice = st.radio(
        "Select neighborhood:",
        ["Mission District", "SoMa (Southern)"],
        horizontal=True,
        key="part2_zone_toggle",
        label_visibility="collapsed",
    )
    zone_label_short = "Mission" if zone_choice == "Mission District" else "SoMa"

    st.markdown(
        f'<div style="font-size: 0.85em; color: {SLATE_500}; '
        f'margin: -8px 0 16px 0;">Showing crime data for the '
        f'<b>{zone_label_short}</b>. Toggle above to switch.</div>',
        unsafe_allow_html=True,
    )

    # Filter to the selected zone
    z = disp[disp["zone"] == zone_choice].copy().reset_index(drop=True)

    def _render_zone_chart(values, ylabel):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if feb_idx_d is not None:
            ax.axvspan(-0.5, feb_idx_d, color=SLATE_200, alpha=0.25, zorder=0)
            ax.axvspan(feb_idx_d, len(months_d) - 0.5, color=RED,
                       alpha=0.07, zorder=0)
        ax.plot(range(len(z)), values,
                color=RED, marker="o", linewidth=2.8, markersize=6,
                zorder=4)
        if feb_idx_d is not None:
            ax.axvline(feb_idx_d, color=SLATE_700, linewidth=1.5,
                       linestyle="--", alpha=0.6)
            y_min, y_max = ax.get_ylim()
            ax.text(feb_idx_d + 0.15, y_min + (y_max - y_min) * 0.08,
                    "Task Force\nlaunches",
                    fontsize=9, ha="left", va="bottom",
                    color=SLATE_700, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35",
                              facecolor="white", edgecolor="none",
                              alpha=0.95))
        ax.set_xticks(range(0, len(months_d), 3))
        ax.set_xticklabels([months_d[i][2:] for i in range(0, len(months_d), 3)],
                           rotation=0, ha="center", fontsize=9)
        style_axes(ax, ylabel=ylabel)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Two charts side-by-side: Total Crime + Drug Offenses ──
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown(
            f'<div style="font-size: 0.85em; color: {SLATE_500}; '
            f'font-weight: 600; text-transform: uppercase; '
            f'letter-spacing: 0.5px; margin-bottom: 4px;">Total Crime &mdash; '
            f'{zone_label_short}</div>',
            unsafe_allow_html=True,
        )
        _render_zone_chart(z["total_crimes"].values,
                           f"Total crimes / month")

    with chart_col2:
        st.markdown(
            f'<div style="font-size: 0.85em; color: {SLATE_500}; '
            f'font-weight: 600; text-transform: uppercase; '
            f'letter-spacing: 0.5px; margin-bottom: 4px;">Drug Offenses &mdash; '
            f'{zone_label_short}</div>',
            unsafe_allow_html=True,
        )
        _render_zone_chart(z["drug_offenses"].values,
                           f"Drug offenses / month")

    # ── Percent-change cards (mirroring Part 1's pattern) ──
    pre = z[z["year_month"] < "2025-02"]
    post = z[z["year_month"] >= "2025-02"]

    def _pct_change(series_pre, series_post):
        a, b = series_pre.mean(), series_post.mean()
        return ((b - a) / a) * 100 if a > 0 else 0

    total_chg = _pct_change(pre["total_crimes"], post["total_crimes"])
    drug_chg = _pct_change(pre["drug_offenses"], post["drug_offenses"])
    dispatch_chg = _pct_change(pre["dispatch_calls"], post["dispatch_calls"])

    st.markdown(
        f'<div style="font-size: 0.78em; color: {SLATE_500}; '
        f'font-weight: 700; letter-spacing: 1.2px; margin: 18px 0 12px 0;">'
        f'CHANGE IN {zone_label_short.upper()} SINCE TASK FORCE'
        f'</div>',
        unsafe_allow_html=True,
    )

    def _zone_pct_card(label, pct):
        accent = RED if pct > 0 else GREEN
        return f"""
        <div style="
            padding: 22px 22px;
            background: white;
            border-left: 4px solid {accent};
            border-radius: 6px;
            box-shadow: 0 1px 2px rgba(15,23,42,0.05);
            height: 100%;
        ">
            <div style="font-size: 0.78em; color: {SLATE_500};
                        font-weight: 600; letter-spacing: 0.5px;
                        text-transform: uppercase; margin-bottom: 6px;">
                {label}
            </div>
            <div style="font-size: 2.8em; font-weight: 800;
                        color: {accent}; line-height: 1.0;">
                {pct:+.1f}%
            </div>
        </div>
        """

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        st.markdown(_zone_pct_card("Total Crime", total_chg),
                    unsafe_allow_html=True)
    with pcol2:
        st.markdown(_zone_pct_card("Drug Offenses", drug_chg),
                    unsafe_allow_html=True)
    with pcol3:
        st.markdown(_zone_pct_card("911 Dispatch Calls", dispatch_chg),
                    unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-size: 0.9rem; color: {SLATE_700}; '
        f'line-height: 1.5; margin-top: 14px;">'
        f'<b>Drug offenses and disorderly conduct are not Part 1 crimes</b> '
        f'&mdash; they are excluded from the official SFPD dashboards cited '
        f'in press conferences. '
        f'<span style="color: {SLATE_500};">Sources: Mission Local, '
        f'SF Standard, GrowSF, SFPD DMACC data.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Standout (left) + Takeaway (right), side-by-side ──
    yes_count = (statements["mentions_displacement"] == "Yes").sum()
    total_statements = len(statements)

    side_left, side_right = st.columns(2)

    with side_left:
        st.markdown(
            f"""
            <div style="
                margin: 18px 0 8px 0;
                padding: 22px 24px;
                background-color: {SLATE_50};
                border-left: 5px solid {SLATE_700};
                border-radius: 10px;
                height: 100%;
            ">
                <div style="
                    font-size: 2.2em;
                    font-weight: 800;
                    color: {SLATE_900};
                    line-height: 1.0;
                    margin-bottom: 10px;
                ">
                    {yes_count} of {total_statements}
                </div>
                <div style="
                    font-size: 0.98em;
                    color: {SLATE_900};
                    line-height: 1.5;
                ">
                    public statements about the task force acknowledge
                    this displacement. <b>Both came reactively</b>, after
                    journalists or supervisors pressed &mdash; never in
                    proactive press releases, the State of the City, or
                    social media.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with side_right:
        st.markdown(
            f"""
            <div style="
                margin: 18px 0 8px 0;
                padding: 22px 24px;
                background-color: {RED}10;
                border-left: 5px solid {RED};
                border-radius: 10px;
                height: 100%;
            ">
                <div style="
                    font-size: 0.78em;
                    color: {RED};
                    font-weight: 700;
                    letter-spacing: 1.2px;
                    margin-bottom: 10px;
                ">
                    TAKEAWAY
                </div>
                <div style="
                    font-size: 0.98em;
                    color: {SLATE_900};
                    line-height: 1.55;
                ">
                    Crime didn't just <i>stop</i> after Feb 2025 &mdash; it
                    <b>relocated</b>. The Mission and SoMa absorbed what
                    downtown pushed out, in categories (drug offenses,
                    dispatch calls) that officials don't include when they
                    cite citywide crime drops.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ══════════════════════════════════════════════
    # THE NARRATIVE GAP — RESIDENT VOICES
    # ══════════════════════════════════════════════

    st.markdown(
        f'<div style="display:inline-block; padding: 4px 12px; '
        f'background-color: {PURPLE}15; color: {PURPLE}; '
        f'border-radius: 20px; font-size: 0.8em; font-weight: 700; '
        f'letter-spacing: 1px; margin-bottom: 6px;">THE NARRATIVE GAP</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### How residents describe the streets they live on")
    st.markdown(
        '<div style="color: ' + SLATE_500 + '; font-size: 1.0em; '
        'margin-bottom: 18px;">'
        "While the official narrative stays silent, the people in the "
        "neighborhoods absorbing the spillover have been telling a different "
        "story all along."
        '</div>',
        unsafe_allow_html=True,
    )

    sentiment = pd.read_csv("data/resident_sentiment.csv")

    # Curated resident quotes — parent, longtime resident, advocate, youth
    curated_speakers = [
        ("Andrew Wickens",
         "We've got kids on this block. You come home from school, "
         "and people are passed out on our porch."),
        ("Unnamed resident",
         "Stop using this neighborhood as a containment zone."),
        ("Reese Isbell",
         "These neighborhoods have basically been put together as "
         "containment zones for the city's problems."),
        ("Ziggy Brown",
         "I grew up with people on the streets all day 24/7"),
    ]

    def find_quote(speaker, partial_quote):
        match = sentiment[
            (sentiment["speaker"] == speaker)
            & sentiment["quote"].str.contains(partial_quote[:25],
                                               regex=False, na=False)
        ]
        if len(match) > 0:
            return match.iloc[0]
        return None

    # Two-column layout for the four curated quotes
    quote_cols = st.columns(2)
    for i, (speaker, partial_q) in enumerate(curated_speakers):
        q = find_quote(speaker, partial_q)
        if q is None:
            continue
        with quote_cols[i % 2]:
            st.markdown(
                f"""
                <div style="
                    background: white;
                    border-left: 3px solid {RED};
                    padding: 14px 18px;
                    margin-bottom: 14px;
                    border-radius: 4px;
                    box-shadow: 0 1px 2px rgba(15,23,42,0.05);
                ">
                    <div style="font-style: italic; color: {SLATE_900};
                                font-size: 0.98em; line-height: 1.5;
                                margin-bottom: 10px;">
                        &ldquo;{q['quote']}&rdquo;
                    </div>
                    <div style="font-size: 0.83em; color: {SLATE_500};">
                        &mdash; <b>{q['speaker']}</b>, {q['identity']}
                        &nbsp;&middot;&nbsp;
                        <a href="{q['url']}" target="_blank"
                           style="color: {SLATE_500};">{q['source']}</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div style="font-size: 0.85em; color: {SLATE_500}; '
        f'font-style: italic; margin-top: 4px;">'
        f'Drawn from Mission Local, SF Standard, ABC7, NBC Bay Area, '
        f'and El Tecolote (2025-2026).'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Closing statement ──
    st.markdown(
        f"""
        <div style="
            margin: 40px 0 16px 0;
            padding: 32px 40px;
            background: linear-gradient(135deg, {PURPLE}18 0%, white 100%);
            border-left: 8px solid {PURPLE};
            border-radius: 14px;
            box-shadow: 0 2px 8px rgba(124, 58, 237, 0.08);
        ">
            <div style="font-size: 0.82em; color: {PURPLE};
                        font-weight: 700; letter-spacing: 1.5px;
                        margin-bottom: 14px;">
                THE TAKEAWAY
            </div>
            <div style="font-size: 1.4em; line-height: 1.45;
                        color: {SLATE_900}; font-weight: 500;">
            When success is measured only in the categories officials choose
            to count, and only in the neighborhoods they choose to celebrate,
            the story they tell is true &mdash; <b>and incomplete</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    next_tab_arrow("Office-to-Residential & DRD", BLUE)


# ══════════════════════════════════════════════
# TAB 3: PURPOSE
# ══════════════════════════════════════════════

with tab3:
    st.markdown(
        f'<div style="text-align:center; padding:64px 20px 32px 20px; '
        f'max-width:760px; margin:0 auto;">'
        f'<div style="font-size:2.6em; font-weight:800; color:{SLATE_900}; '
        f'line-height:1.2; margin-bottom:32px; letter-spacing:-0.01em;">'
        f"Policies don&apos;t speak for themselves."
        f'<br>'
        f'<span style="background:linear-gradient(180deg, transparent 62%, '
        f'{PURPLE}33 62%); padding:0 6px;">'
        f"The data behind them does."
        f'</span>'
        f'</div>'
        f'<div style="font-size:1.1em; color:{SLATE_700}; line-height:1.75; '
        f'max-width:620px; margin:0 auto;">'
        f"Voter education and policy advocacy are how the gap between "
        f'<b style="color:{SLATE_900};">what officials say</b> and '
        f'<b style="color:{PURPLE};">what the data shows</b> gets closed. '
        f"This dashboard is one tool &mdash; read past press releases, "
        f"ask harder questions, and vote on what the data reveals, not "
        f"what the narrative promises."
        f'</div>'
        f'<div style="margin-top:48px; display:flex; justify-content:center; '
        f'align-items:center; gap:14px;">'
        f'<div style="height:1px; width:48px; background:{SLATE_200};"></div>'
        f'<div style="font-size:0.72em; color:{SLATE_500}; '
        f'font-weight:700; letter-spacing:2.5px;">'
        f'BRINGING SF BACK?'
        f'</div>'
        f'<div style="height:1px; width:48px; background:{SLATE_200};"></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

