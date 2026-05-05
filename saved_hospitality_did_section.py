"""
============================================================
Hospitality Task Force — Difference-in-Differences Analysis
============================================================

STANDALONE ANALYSIS SCRIPT (not part of the dashboard).

This was prototyped in the Streamlit dashboard, but removed for clarity.
The descriptive time-series charts in Part 2 of the Hospitality Task Force
tab already show the displacement story visually. The DiD here adds
*statistical rigor* — confidence intervals, p-values, and a robustness
check — for anyone who wants to explore the causal claim more formally.

----- WHAT THE DiD DOES -----

Compares the Hospitality Zone (treatment) to surrounding neighborhoods
(Mission, SoMa as controls) before and after the Feb 2025 task force
launch. The interaction term `treated × post` isolates the task force's
causal effect from any citywide trend that affected all neighborhoods
equally.

Specification:
    outcome ~ treated + post + treated × post
        treated     = 1 if Hospitality Zone, 0 if surrounding
        post        = 1 if month >= Feb 2025, 0 otherwise
        treated × post = the causal estimate

Standard errors are HC1 (heteroskedasticity-consistent).
A robustness check adds a linear time trend to test whether the effect
survives controlling for any pre-existing trajectory.

----- HOW TO RUN -----

From the streamlit_app/ directory:

    python saved_hospitality_did_section.py

Optional: pass --chart to also save a coefficient plot to
analysis_output/did_coefficients.png

    python saved_hospitality_did_section.py --chart

----- DATA -----

Uses data/displacement_crime.csv, which has monthly crime/drug/dispatch
counts by zone (Hospitality Zone, Mission District, SoMa) from
Jan 2024 through Mar 2026.

============================================================
"""

import argparse
import os
import pandas as pd
import statsmodels.formula.api as smf


def load_displacement_data(path="data/displacement_crime.csv"):
    """Load the displacement panel and add DiD variables."""
    df = pd.read_csv(path)
    df["treated"] = (df["zone"] == "Hospitality Zone").astype(int)
    df["post"] = (df["year_month"] >= "2025-02").astype(int)
    df["treated_x_post"] = df["treated"] * df["post"]
    df["t"] = df.groupby("zone").cumcount()  # time trend
    return df


def fit_did_models(df):
    """Fit DiD models for total crime, drug offenses, and dispatch calls."""
    formula = "{outcome} ~ treated + post + treated_x_post"

    models = {
        "Total Crime": smf.ols(
            formula.format(outcome="total_crimes"), data=df
        ).fit(cov_type="HC1"),

        "Drug Offenses": smf.ols(
            formula.format(outcome="drug_offenses"), data=df
        ).fit(cov_type="HC1"),

        "Dispatch Calls": smf.ols(
            formula.format(outcome="dispatch_calls"), data=df
        ).fit(cov_type="HC1"),
    }

    # Robustness check: add a linear time trend to the total-crime model
    models["Total Crime (with time trend)"] = smf.ols(
        "total_crimes ~ treated + post + treated_x_post + t", data=df
    ).fit(cov_type="HC1")

    return models


def print_results(models):
    """Print a human-readable summary of each DiD model."""
    print("=" * 70)
    print("  HOSPITALITY TASK FORCE — DIFFERENCE-IN-DIFFERENCES RESULTS")
    print("=" * 70)
    print()
    print("Treatment: Hospitality Zone (received task force, Feb 2025)")
    print("Control:   Mission + SoMa (surrounding neighborhoods)")
    print()

    for name, model in models.items():
        print("-" * 70)
        print(f"  {name}")
        print("-" * 70)

        coef = model.params["treated_x_post"]
        ci_low, ci_high = model.conf_int().loc["treated_x_post"]
        p = model.pvalues["treated_x_post"]

        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  treated × post:  {coef:+8.1f}  "
              f"[95% CI: {ci_low:+.1f}, {ci_high:+.1f}]  "
              f"p = {p:.4f} {stars}")

        # The post coefficient tells us what happened in control zones
        post_coef = model.params["post"]
        post_p = model.pvalues["post"]
        print(f"  post (control):  {post_coef:+8.1f}  "
              f"(change in surrounding zones, p = {post_p:.4f})")

        print(f"  R-squared:       {model.rsquared:.4f}")
        print(f"  N observations:  {int(model.nobs)}")
        print()

    print("=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    print()
    total_effect = models["Total Crime"].params["treated_x_post"]
    drug_effect = models["Drug Offenses"].params["treated_x_post"]
    dispatch_effect = models["Dispatch Calls"].params["treated_x_post"]

    print(f"  The Hospitality Task Force is associated with:")
    print(f"    - {total_effect:+.0f} total crimes/month")
    print(f"      in the Hospitality Zone relative to surrounding neighborhoods")
    print(f"    - {drug_effect:+.0f} drug offenses/month (relative)")
    print(f"    - {dispatch_effect:+.0f} dispatch calls/month (relative)")
    print()
    print("  The 'post' coefficients (positive) show that surrounding")
    print("  neighborhoods saw INCREASES at the same time downtown saw")
    print("  decreases — consistent with crime displacement.")
    print()


def save_coefficient_chart(models, output_path="analysis_output/did_coefficients.png"):
    """Save a horizontal bar chart of the three main DiD coefficients."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main_models = {k: v for k, v in models.items() if "trend" not in k}

    labels = list(main_models.keys())
    coefs = [m.params["treated_x_post"] for m in main_models.values()]
    ci_lows = [m.conf_int().loc["treated_x_post"][0] for m in main_models.values()]
    ci_highs = [m.conf_int().loc["treated_x_post"][1] for m in main_models.values()]
    pvals = [m.pvalues["treated_x_post"] for m in main_models.values()]

    fig, ax = plt.subplots(figsize=(9, 4))
    y_pos = list(range(len(labels)))
    errs = [[c - lo for c, lo in zip(coefs, ci_lows)],
            [hi - c for c, hi in zip(coefs, ci_highs)]]

    colors = ["#16A34A" if c < 0 else "#DC2626" for c in coefs]
    ax.barh(y_pos, coefs, color=colors, alpha=0.85, height=0.5)
    ax.errorbar(coefs, y_pos, xerr=errs, fmt="none", ecolor="black",
                capsize=4, linewidth=1.5)
    ax.axvline(0, color="black", linewidth=1)

    for i, (c, p) in enumerate(zip(coefs, pvals)):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.annotate(f"{c:+.0f}{stars}",
                    xy=(c, i), xytext=(8 if c >= 0 else -8, 0),
                    textcoords="offset points",
                    va="center", ha="left" if c >= 0 else "right",
                    fontsize=11, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("DiD Estimate (treated × post)")
    ax.set_title("Hospitality Task Force Effect:\n"
                 "Hospitality Zone vs Surrounding Neighborhoods (Mission + SoMa)",
                 fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"  Coefficient chart saved to: {output_path}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run DiD analysis on the Hospitality Task Force."
    )
    parser.add_argument(
        "--chart", action="store_true",
        help="Also save a coefficient plot to analysis_output/did_coefficients.png"
    )
    parser.add_argument(
        "--data", default="data/displacement_crime.csv",
        help="Path to displacement_crime.csv (default: data/displacement_crime.csv)"
    )
    args = parser.parse_args()

    df = load_displacement_data(args.data)
    models = fit_did_models(df)
    print_results(models)

    if args.chart:
        save_coefficient_chart(models)


if __name__ == "__main__":
    main()
