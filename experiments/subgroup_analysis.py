"""
RQ4: does quality change by subgroup?
breaks coverage and width down by season, peak/off-peak, line and cause.
python -m experiments.subgroup_analysis
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.model import DelayRangeEstimator, DELAY_CAP_MIN

CLEANED = Path("data/processed/cleaned.csv")
OUT = Path("results")
SEASON = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
          6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}


def summarize(te, lo, hi):
    y = te["min_delay"].to_numpy().astype(float)
    te = te.copy()
    te["_in"] = (y >= lo) & (y <= hi)
    te["_w"] = hi - lo

    def grp(col):
        #coverage + width per group, drop tiny groups
        g = te.groupby(col)
        out = g.agg(n=("_in", "size"), coverage=("_in", "mean"), median_width=("_w", "median"))
        out["coverage"] = out["coverage"].round(3); out["median_width"] = out["median_width"].round(1)
        return out[out["n"] >= 30].sort_values("median_width", ascending=False)

    hour = pd.to_datetime(te["time"].astype(str), format="%H:%M", errors="coerce").dt.hour.fillna(0)
    te["season"] = pd.to_datetime(te["date"], errors="coerce").dt.month.map(SEASON)
    te["period"] = np.where(hour.isin([7, 8, 9, 16, 17, 18]), "peak", "off-peak")
    return grp("season"), grp("period"), grp("line"), grp("code")


def main():
    df = pd.read_csv(CLEANED)
    df = df[df["min_delay"] > 0].copy()
    df["min_delay"] = df["min_delay"].clip(upper=DELAY_CAP_MIN)
    df = df.assign(_d=pd.to_datetime(df["date"], errors="coerce")).sort_values("_d").reset_index(drop=True)
    n = len(df); tr, cal, te = df.iloc[:int(.7*n)], df.iloc[int(.7*n):int(.8*n)], df.iloc[int(.8*n):]
    est = DelayRangeEstimator().fit(tr).conformalize(cal)
    lo, mid, hi = est.predict_range(te)

    season, period, line, code = summarize(te, lo, hi)
    pd.set_option("display.width", 200)
    print("=== RQ4: coverage & width by subgroup (future fold, nominal 90%) ===")
    print("\n-- By season (sorted by width = uncertainty) --\n" + season.to_string())
    print("\n-- Peak vs off-peak --\n" + period.to_string())
    print("\n-- By line --\n" + line.to_string())
    print("\n-- By delay cause (top by width; >=30 incidents) --\n" + code.head(8).to_string())
    print("\n-- Most RELIABLE causes (narrowest, well-covered) --\n" +
          code[code.coverage >= 0.85].sort_values("median_width").head(6).to_string())

    for name, t in [("season", season), ("period", period), ("line", line), ("code", code)]:
        t.to_csv(OUT / f"subgroup_{name}.csv")
    print(f"\nSaved subgroup_*.csv -> {OUT}/")


if __name__ == "__main__":
    main()
