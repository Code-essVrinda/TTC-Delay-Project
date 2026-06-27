"""
RQ1: which features matter?
add feature groups one by one and check the 90% interval.
python -m experiments.ablation
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.model import features, q_model, conformal_Q, DELAY_CAP_MIN

CLEANED = Path("data/processed/cleaned.csv")
OUT = Path("results")
LO_Q, HI_Q = 0.05, 0.95
ALPHA = LO_Q + (1 - HI_Q)

TIME = ["hour", "dow", "month", "is_weekend", "is_rush", "hour_sin", "hour_cos"]
LEVELS = [
    ("time only", TIME),
    ("+ station", TIME + ["station_mean", "station_hour_mean"]),
    ("+ line", TIME + ["station_mean", "station_hour_mean", "line_mean"]),
    ("+ delay code", TIME + ["station_mean", "station_hour_mean", "line_mean", "code_mean"]),
    ("+ direction (full)", TIME + ["station_mean", "station_hour_mean", "line_mean", "code_mean", "bound_mean"]),
]


def winkler(y, lo, hi, a):
    return float(np.mean((hi - lo) + (2/a)*np.maximum(0, lo-y) + (2/a)*np.maximum(0, y-hi)))


def main():
    df = pd.read_csv(CLEANED)
    df = df[df["min_delay"] > 0].copy()
    df["min_delay"] = df["min_delay"].clip(upper=DELAY_CAP_MIN)
    df = df.assign(_d=pd.to_datetime(df["date"], errors="coerce")).sort_values("_d").reset_index(drop=True)
    n = len(df); tr, cal, te = df.iloc[:int(.7*n)], df.iloc[int(.7*n):int(.8*n)], df.iloc[int(.8*n):]
    ytr, ycal, yte = [x["min_delay"].to_numpy().astype(float) for x in (tr, cal, te)]
    Xtr, maps, gm = features(tr, fit=True, y=ytr)
    Xcal, Xte = features(cal, maps, gm), features(te, maps, gm)

    rows = []
    for name, cols in LEVELS:
        #train lo/hi on this feature subset, conformalize, score
        m_lo = q_model(LO_Q).fit(Xtr[cols], ytr)
        m_hi = q_model(HI_Q).fit(Xtr[cols], ytr)
        Q = conformal_Q(m_lo.predict(Xcal[cols]), m_hi.predict(Xcal[cols]), ycal, ALPHA)
        lo = np.clip(m_lo.predict(Xte[cols]) - Q, 0, None); hi = m_hi.predict(Xte[cols]) + Q
        rows.append({"feature_set": name, "n_features": len(cols),
                     "coverage": round(float(np.mean((yte >= lo) & (yte <= hi))), 3),
                     "median_width": round(float(np.median(hi - lo)), 1),
                     "winkler": round(winkler(yte, lo, hi, ALPHA), 2)})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("=== RQ1: Ablation — feature groups vs 90% interval quality (future fold) ===\n")
    print(out.to_string(index=False))
    out.to_csv(OUT / "ablation.csv", index=False)
    print(f"\nSaved -> {OUT/'ablation.csv'}")


if __name__ == "__main__":
    main()
