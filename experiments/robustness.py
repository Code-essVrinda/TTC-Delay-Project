"""
RQ4: robustness.
(A) try 80/85/90/95% intervals.
(B) drift: train on 2023 only, test on later years, watch coverage drop.
python -m experiments.robustness
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.model import features, q_model, conformal_Q, DELAY_CAP_MIN

CLEANED = Path("data/processed/cleaned.csv")
OUT = Path("results")


def load():
    df = pd.read_csv(CLEANED)
    df = df[df["min_delay"] > 0].copy()
    df["min_delay"] = df["min_delay"].clip(upper=DELAY_CAP_MIN)
    return df.assign(_d=pd.to_datetime(df["date"], errors="coerce")).sort_values("_d").reset_index(drop=True)


def fit_cqr(tr, cal, lo_q, hi_q):
    ytr = tr["min_delay"].to_numpy().astype(float)
    ycal = cal["min_delay"].to_numpy().astype(float)
    Xtr, maps, gm = features(tr, fit=True, y=ytr)
    m_lo = q_model(lo_q).fit(Xtr, ytr); m_hi = q_model(hi_q).fit(Xtr, ytr)
    Q = conformal_Q(m_lo.predict(features(cal, maps, gm)), m_hi.predict(features(cal, maps, gm)),
                    ycal, lo_q + (1 - hi_q))
    return m_lo, m_hi, Q, maps, gm


def interval(model, df):
    m_lo, m_hi, Q, maps, gm = model
    X = features(df, maps, gm)
    return np.clip(m_lo.predict(X) - Q, 0, None), m_hi.predict(X) + Q


def main():
    df = load()
    n = len(df); tr, cal, te = df.iloc[:int(.7*n)], df.iloc[int(.7*n):int(.8*n)], df.iloc[int(.8*n):]
    yte = te["min_delay"].to_numpy().astype(float)

    #(A) different interval levels
    print("=== RQ4a: robustness across nominal interval levels (future fold) ===\n")
    rowsA = []
    for nominal, (lo_q, hi_q) in [(80, (.10, .90)), (85, (.075, .925)), (90, (.05, .95)), (95, (.025, .975))]:
        mdl = fit_cqr(tr, cal, lo_q, hi_q)
        lo, hi = interval(mdl, te)
        rowsA.append({"nominal_%": nominal, "empirical_coverage": round(float(np.mean((yte >= lo) & (yte <= hi))), 3),
                      "median_width_min": round(float(np.median(hi - lo)), 1)})
    print(pd.DataFrame(rowsA).to_string(index=False))

    #(B) train on 2023, test on each later year
    print("\n=== RQ4b: coverage under temporal drift (trained on 2023 only, nominal 90%) ===\n")
    d23 = df[df["year"] == 2023]
    n23 = len(d23); tr23, cal23 = d23.iloc[:int(.7*n23)], d23.iloc[int(.7*n23):int(.85*n23)]
    te23 = d23.iloc[int(.85*n23):]
    mdl = fit_cqr(tr23, cal23, 0.05, 0.95)
    rowsB = []
    test_sets = [("2023 (held-out)", te23)] + [(str(y), df[df["year"] == y]) for y in (2024, 2025, 2026)]
    for label, ds in test_sets:
        if len(ds) < 30:
            continue
        y = ds["min_delay"].to_numpy().astype(float)
        lo, hi = interval(mdl, ds)
        rowsB.append({"test_year": label, "n": len(ds),
                      "coverage": round(float(np.mean((y >= lo) & (y <= hi))), 3),
                      "median_width_min": round(float(np.median(hi - lo)), 1)})
    dfB = pd.DataFrame(rowsB)
    print(dfB.to_string(index=False))

    pd.DataFrame(rowsA).to_csv(OUT / "robustness_levels.csv", index=False)
    dfB.to_csv(OUT / "robustness_drift.csv", index=False)
    print(f"\nSaved -> {OUT/'robustness_levels.csv'}, {OUT/'robustness_drift.csv'}")


if __name__ == "__main__":
    main()
