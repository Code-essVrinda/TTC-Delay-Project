"""
RQ2: does conformal prediction actually help?
compares 4 ways to make a 90% interval, same data and features:
  B1 median + residual, B2 RF quantile, B3 lgbm quantile (no conformal),
  B4 CQR (ours). also climatology. with bootstrap CIs.
python -m experiments.baseline_models
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from quantile_forest import RandomForestQuantileRegressor

from src.model import features, q_model, conformal_Q, DELAY_CAP_MIN

CLEANED = Path("data/processed/cleaned.csv")
OUT = Path("results")
LO_Q, HI_Q = 0.05, 0.95
ALPHA = LO_Q + (1 - HI_Q)          #0.10 -> 90% interval
NBOOT = 1000


def winkler(y, lo, hi, alpha):
    return (hi - lo) + (2 / alpha) * np.maximum(0, lo - y) + (2 / alpha) * np.maximum(0, y - hi)


def load_split():
    df = pd.read_csv(CLEANED)
    df = df[df["min_delay"] > 0].copy()
    df["min_delay"] = df["min_delay"].clip(upper=DELAY_CAP_MIN)
    df = df.assign(_d=pd.to_datetime(df["date"], errors="coerce")).sort_values("_d").reset_index(drop=True)
    n = len(df)
    return df.iloc[:int(.7*n)], df.iloc[int(.7*n):int(.8*n)], df.iloc[int(.8*n):]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr, cal, te = load_split()
    ytr = tr["min_delay"].to_numpy().astype(float)
    ycal = cal["min_delay"].to_numpy().astype(float)
    yte = te["min_delay"].to_numpy().astype(float)
    Xtr, maps, gm = features(tr, fit=True, y=ytr)
    Xcal, Xte = features(cal, maps, gm), features(te, maps, gm)

    intervals = {}   #name -> (lo, hi)

    #B1 median + a fixed residual interval
    med = q_model(0.50).fit(Xtr, ytr)
    res = ycal - med.predict(Xcal)
    d_lo, d_hi = np.quantile(res, LO_Q), np.quantile(res, HI_Q)
    intervals["B1 Median + residual"] = (np.clip(med.predict(Xte) + d_lo, 0, None), med.predict(Xte) + d_hi)

    #B2 random forest quantile, raw
    qrf = RandomForestQuantileRegressor(n_estimators=300, min_samples_leaf=20, random_state=42)
    qrf.fit(Xtr.values, ytr)
    qp = qrf.predict(Xte.values, quantiles=[LO_Q, HI_Q])
    intervals["B2 RF-Quantile (raw)"] = (np.clip(qp[:, 0], 0, None), qp[:, 1])

    #B3 lgbm quantile, no conformal
    m_lo, m_hi = q_model(LO_Q).fit(Xtr, ytr), q_model(HI_Q).fit(Xtr, ytr)
    lo3, hi3 = m_lo.predict(Xte), m_hi.predict(Xte)
    intervals["B3 LGBM-Quantile (raw)"] = (np.clip(lo3, 0, None), hi3)

    #B4 ours = B3 + conformal
    Q = conformal_Q(m_lo.predict(Xcal), m_hi.predict(Xcal), ycal, ALPHA)
    intervals["B4 CQR (ours)"] = (np.clip(lo3 - Q, 0, None), hi3 + Q)

    #climatology
    g_lo, g_hi = np.quantile(ytr, LO_Q), np.quantile(ytr, HI_Q)
    Qg = conformal_Q(np.full(len(ycal), g_lo), np.full(len(ycal), g_hi), ycal, ALPHA)
    intervals["Climatology"] = (np.full(len(yte), max(0, g_lo - Qg)), np.full(len(yte), g_hi + Qg))

    #metrics + bootstrap CIs. same resampled rows for every method
    rng = np.random.default_rng(42)
    n = len(yte)
    boot_idx = [rng.integers(0, n, n) for _ in range(NBOOT)]

    rows, wink_boot = [], {}
    for name, (lo, hi) in intervals.items():
        w = winkler(yte, lo, hi, ALPHA)
        cov = (yte >= lo) & (yte <= hi)
        wb = np.array([w[ix].mean() for ix in boot_idx])
        cb = np.array([cov[ix].mean() for ix in boot_idx])
        wink_boot[name] = wb
        rows.append({
            "method": name,
            "coverage": round(cov.mean(), 3),
            "coverage_95CI": f"[{np.quantile(cb,.025):.3f}, {np.quantile(cb,.975):.3f}]",
            "median_width": round(float(np.median(hi - lo)), 1),
            "winkler": round(float(w.mean()), 2),
            "winkler_95CI": f"[{np.quantile(wb,.025):.2f}, {np.quantile(wb,.975):.2f}]",
        })
    res_df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== RQ2: 90% prediction-interval methods (held-out future fold) ===\n")
    print(res_df.to_string(index=False))
    res_df.to_csv(OUT / "baseline_comparison.csv", index=False)

    #paired bootstrap so we can put a CI on the improvement
    def improvement(a, b):
        d = 100 * (1 - wink_boot[a] / wink_boot[b])
        return d.mean(), np.quantile(d, .025), np.quantile(d, .975)

    print("\n=== Isolating the value of conformalization & the model ===")
    for a, b, label in [("B4 CQR (ours)", "B3 LGBM-Quantile (raw)", "conformalization (B4 vs B3)"),
                        ("B4 CQR (ours)", "Climatology", "full model vs climatology"),
                        ("B4 CQR (ours)", "B2 RF-Quantile (raw)", "vs RF-Quantile")]:
        m, lo, hi = improvement(a, b)
        print(f"  Winkler improvement, {label:32}: {m:5.1f}%   95% CI [{lo:.1f}%, {hi:.1f}%]")

    c3 = res_df.loc[res_df.method == "B3 LGBM-Quantile (raw)", "coverage"].iloc[0]
    c4 = res_df.loc[res_df.method == "B4 CQR (ours)", "coverage"].iloc[0]
    print(f"\n  Coverage: raw quantiles (B3) = {c3:.3f}  ->  conformalized (B4) = {c4:.3f}  (nominal 0.90)")
    print(f"  Saved -> {OUT/'baseline_comparison.csv'}")


if __name__ == "__main__":
    main()
