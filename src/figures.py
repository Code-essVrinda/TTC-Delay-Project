"""
makes all the figures for the readme and report, plus prints the case studies.
some figures use the model, some read the result csvs. so run this AFTER the
model and the experiments. saves to results/figures/.
python -m src.figures
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model import DelayRangeEstimator, DELAY_CAP_MIN

CLEANED = Path("data/processed/cleaned.csv")
RESULTS = Path("results")
FIG = Path("results/figures")
BLUE, DARK, GREEN, RED, GREY, ORANGE = "#4C72B0", "#1F3A68", "#2CA02C", "#D62728", "#999999", "#DD8452"


def load_split():
    df = pd.read_csv(CLEANED)
    df = df[df["min_delay"] > 0].copy()
    df["min_delay"] = df["min_delay"].clip(upper=DELAY_CAP_MIN)
    df = df.assign(_d=pd.to_datetime(df["date"], errors="coerce")).sort_values("_d").reset_index(drop=True)
    n = len(df)
    return df.iloc[:int(.7*n)], df.iloc[int(.7*n):int(.8*n)], df.iloc[int(.8*n):]


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=130); plt.close(fig)


#figures that use the model / data

def fig_distribution(df_all):
    #histogram of delay length
    y = df_all["min_delay"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(y, bins=range(0, 62, 2), color=BLUE, edgecolor="white")
    ax.axvline(np.median(y), color=DARK, ls="--", lw=2, label=f"median = {np.median(y):.0f} min")
    ax.axvline(60, color=RED, ls=":", lw=2, label="capped at 60 min (major disruption)")
    ax.set_title("Most subway delays are short — but the tail is long", fontsize=14, fontweight="bold")
    ax.set_xlabel("Delay duration (minutes)"); ax.set_ylabel("Number of incidents")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "delay_distribution.png")


def fig_examples(model, te):
    #range vs actual for a few incidents
    rng = te.iloc[::max(1, len(te)//400)].head(14).reset_index(drop=True)
    lo, mid, hi = model.predict_range(rng)
    actual = rng["min_delay"].to_numpy()
    order = np.argsort(mid)
    lo, mid, hi, actual = lo[order], mid[order], hi[order], actual[order]
    labels = [f"{r.station[:14]} ({r.code})" for r in rng.iloc[order].itertuples()]
    inside = (actual >= lo) & (actual <= hi)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    yy = np.arange(len(rng))
    ax.hlines(yy, lo, hi, color=BLUE, lw=7, alpha=0.45, label="predicted range (9 of 10)")
    ax.plot(mid, yy, "o", color=DARK, ms=6, label="most-likely guess")
    ax.scatter(actual[inside], yy[inside], marker="D", color=GREEN, s=55, zorder=5, label="actual (inside range ✓)")
    ax.scatter(actual[~inside], yy[~inside], marker="X", color=RED, s=80, zorder=5, label="actual (outside ✗)")
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Delay (minutes)")
    ax.set_title("Predicted range vs. what actually happened", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.02)
    save(fig, "example_predictions.png")


def fig_coverage_width(tr, cal, te):
    #more right = wider range
    yte = te["min_delay"].to_numpy().astype(float)
    rights, widths, labels = [], [], []
    for lo_q, hi_q, lab in [(0.10, 0.90, "aim 8/10"), (0.05, 0.95, "aim 9/10"), (0.025, 0.975, "aim 9.5/10")]:
        e = DelayRangeEstimator(lo_q, hi_q).fit(tr).conformalize(cal)
        lo, mid, hi = e.predict_range(te)
        rights.append(np.mean((yte >= lo) & (yte <= hi)) * 10)
        widths.append(np.median(hi - lo)); labels.append(lab)
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(8, 5))
    b = ax1.bar(x - 0.2, rights, 0.4, color=GREEN, label="how often right (out of 10)")
    ax1.set_ylabel("Right out of 10", color=GREEN); ax1.set_ylim(0, 10.5)
    ax1.bar_label(b, fmt="%.1f", padding=2, color=GREEN, fontweight="bold")
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + 0.2, widths, 0.4, color=BLUE, label="range width (min)")
    ax2.set_ylabel("Range width (minutes)", color=BLUE)
    ax2.bar_label(b2, fmt="%.0f min", padding=2, color=DARK, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_title("The trustworthiness knob: more often right = wider range", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "coverage_vs_width.png", dpi=130); plt.close(fig)


def fig_drivers(df_all):
    #codes that show up a lot in long delays
    band = df_all[(df_all.min_delay >= 30) & (df_all.min_delay <= 60)]
    base = df_all["code"].value_counts(normalize=True)
    inb = band["code"].value_counts(normalize=True)
    cnt = band["code"].value_counts()
    lift = (inb / base).replace([np.inf], np.nan).dropna()
    lift = lift[cnt.reindex(lift.index) >= 8].sort_values(ascending=False).head(7)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(lift.index, lift.values, color=BLUE, edgecolor="white")
    for i, v in enumerate(lift.values):
        ax.text(v + 0.3, i, f"{v:.0f}x  (~{min(99, v*1.8):.0f}% run long)", va="center", fontsize=9, color=DARK)
    ax.set_xlabel("How much more likely than a normal delay (lift)")
    ax.set_title("Long delays are driven by the REASON, not the place", fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False); ax.set_xlim(0, lift.max() * 1.35)
    save(fig, "long_delay_drivers.png")


#figures that read the result csvs (run the experiments first)

def fig_baselines():
    d = pd.read_csv(RESULTS / "baseline_comparison.csv")
    d["short"] = d["method"].str.replace(r"\s*\(.*\)", "", regex=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.barh(d["short"][::-1], d["coverage"][::-1], color=BLUE); a1.axvline(0.90, color=RED, ls="--", label="nominal 0.90")
    a1.set_title("Empirical coverage", fontweight="bold"); a1.set_xlim(0.8, 1.0); a1.legend(fontsize=8)
    a2.barh(d["short"][::-1], d["winkler"][::-1], color=DARK)
    a2.set_title("Winkler interval score (lower better)", fontweight="bold")
    fig.suptitle("Method comparison: conformalization restores calibration", fontsize=13, fontweight="bold")
    save(fig, "baselines.png")


def fig_ablation():
    d = pd.read_csv(RESULTS / "ablation.csv")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(d["feature_set"], d["winkler"], "o-", color=DARK, lw=2, ms=8)
    for x, y in zip(d["feature_set"], d["winkler"]):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontweight="bold")
    ax.set_ylabel("Winkler score (lower better)"); ax.set_title("Ablation: the delay code carries the signal", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=15); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "ablation.png")


def fig_importance():
    d = pd.read_csv(RESULTS / "feature_importance.csv").head(8)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(d["feature"], d["importance"], color=BLUE, edgecolor="white")
    ax.set_xlabel("Permutation importance (MAE increase when shuffled)")
    ax.set_title("Feature importance: cause dominates, predictively", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "feature_importance.png")


def fig_calibration():
    d = pd.read_csv(RESULTS / "robustness_levels.csv")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.8, 0.95], [0.8, 0.95], "k--", label="perfect calibration")
    ax.plot(d["nominal_%"] / 100, d["empirical_coverage"], "o-", color=GREEN, ms=9, lw=2, label="achieved")
    for _, r in d.iterrows():
        ax.annotate(f"{r.median_width_min:.0f} min wide", (r['nominal_%']/100, r.empirical_coverage),
                    textcoords="offset points", xytext=(8, -12), fontsize=8)
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration across interval levels", fontsize=13, fontweight="bold"); ax.legend()
    save(fig, "calibration.png")


def fig_drift():
    d = pd.read_csv(RESULTS / "robustness_drift.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(d["test_year"], d["coverage"], "o-", color=ORANGE, lw=2, ms=9)
    ax.axhline(0.90, color=RED, ls="--", label="nominal 0.90")
    ax.set_ylim(0.8, 0.95); ax.set_ylabel("Coverage"); ax.legend()
    ax.set_title("Temporal drift: trained on 2023, tested on later years", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "drift.png")


def fig_uncertainty():
    d = pd.read_csv(RESULTS / "uncertainty_prediction.csv")
    x = np.arange(len(d)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(x - w/2, d["mean_actual_error"], w, color=RED, label="actual error (min)")
    ax.bar(x + w/2, d["mean_interval_width"], w, color=BLUE, label="interval width (min)")
    ax.set_xticks(x); ax.set_xticklabels(d["band"]); ax.set_xlabel("Predicted-uncertainty tercile")
    ax.set_ylabel("Minutes"); ax.legend()
    ax.set_title("Uncertainty is predictable: error & width rise together", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "uncertainty_terciles.png")


def fig_cause():
    d = pd.read_csv(RESULTS / "subgroup_code.csv")
    top = d.sort_values("median_width", ascending=False).head(6)
    bot = d[d.coverage >= 0.85].sort_values("median_width").head(6)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["code"], top["median_width"], color=RED, label="most uncertain causes")
    ax.barh(bot["code"], bot["median_width"], color=GREEN, label="most reliable causes")
    ax.set_xlabel("Median interval width (min) — uncertainty by delay cause")
    ax.set_title("Uncertainty is driven by the delay cause", fontsize=13, fontweight="bold")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "cause_uncertainty.png")


def case_studies(te, model):
    #pick 3 incidents: confident+right, a miss, a very wide one
    lo, mid, hi = model.predict_range(te)
    y = te["min_delay"].to_numpy(); w = hi - lo; inside = (y >= lo) & (y <= hi)
    pick = {
        "Confident & correct": int(np.argmin(np.where(inside & (w <= 4), w + np.abs(y-mid), 1e9))),
        "Missed (rare outcome)": int(np.argmax(np.where(~inside, y - hi, -1e9))),
        "High uncertainty (wide)": int(np.argmax(w)),
    }
    print("\n=== Prediction case studies (future fold) ===")
    for label, i in pick.items():
        r = te.iloc[i]
        print(f"  {label:24} | {r.station} / {r.code} | predicted {mid[i]} min, range {lo[i]}-{hi[i]} | actual {int(y[i])} min")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    tr, cal, te = load_split()
    te = te.reset_index(drop=True)
    all_real = pd.concat([tr, cal, te])
    model = DelayRangeEstimator.load()

    #model/data figures
    fig_distribution(all_real)
    fig_examples(model, te)
    fig_coverage_width(tr, cal, te)
    fig_drivers(all_real)

    #csv figures (need the experiments to have run)
    try:
        fig_baselines(); fig_ablation(); fig_importance()
        fig_calibration(); fig_drift(); fig_uncertainty(); fig_cause()
        n_csv = 7
    except FileNotFoundError as e:
        print(f"(skipped some figures — run the experiments first: {e})")
        n_csv = 0
    print(f"Saved {4 + n_csv} figures to {FIG}/")
    case_studies(te, model)


if __name__ == "__main__":
    main()
