"""
RQ1/RQ3: which features drive the prediction?
permutation importance on the median model. used instead of SHAP (not installed).
python -m experiments.importance
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.model import features, q_model, DELAY_CAP_MIN

CLEANED = Path("data/processed/cleaned.csv")
OUT = Path("results")

#nicer names for the table
NICE = {"code_mean": "Delay code (reason)", "station_hour_mean": "Station × hour",
        "station_mean": "Station", "hour": "Hour of day", "hour_sin": "Hour (cyclic)",
        "hour_cos": "Hour (cyclic)", "line_mean": "Line", "bound_mean": "Direction",
        "month": "Month", "dow": "Day of week", "is_rush": "Rush hour", "is_weekend": "Weekend"}


def main():
    df = pd.read_csv(CLEANED)
    df = df[df["min_delay"] > 0].copy()
    df["min_delay"] = df["min_delay"].clip(upper=DELAY_CAP_MIN)
    df = df.assign(_d=pd.to_datetime(df["date"], errors="coerce")).sort_values("_d").reset_index(drop=True)
    n = len(df); tr, te = df.iloc[:int(.8*n)], df.iloc[int(.8*n):]
    ytr = tr["min_delay"].to_numpy().astype(float)
    yte = te["min_delay"].to_numpy().astype(float)
    Xtr, maps, gm = features(tr, fit=True, y=ytr)
    Xte = features(te, maps, gm)

    med = q_model(0.50).fit(Xtr, ytr)
    r = permutation_importance(med, Xte, yte, scoring="neg_mean_absolute_error",
                               n_repeats=10, random_state=42)
    #group the two hour-cyclic cols under one name, then sort
    imp = pd.DataFrame({"feature": [NICE.get(c, c) for c in Xte.columns],
                        "importance": r.importances_mean, "std": r.importances_std})
    imp = imp.groupby("feature", as_index=False).agg({"importance": "sum", "std": "max"})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["importance"] = imp["importance"].round(3); imp["std"] = imp["std"].round(3)
    pd.set_option("display.width", 200)
    print("=== RQ1/RQ3: Permutation importance (MAE increase when shuffled), median model ===\n")
    print(imp.to_string(index=False))
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    print(f"\nSaved -> {OUT/'feature_importance.csv'}")


if __name__ == "__main__":
    main()
