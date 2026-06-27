"""
RQ3: can we tell in advance where the model will be uncertain?
train a 2nd model to predict the median model's |error| from features.
if it works, uncertainty is learnable. report R2, spearman, AUC, terciles.
python -m experiments.uncertainty_prediction
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score

from src.model import features, q_model, conformal_Q, DELAY_CAP_MIN

CLEANED = Path("data/processed/cleaned.csv")
OUT = Path("results")


def main():
    df = pd.read_csv(CLEANED)
    df = df[df["min_delay"] > 0].copy()
    df["min_delay"] = df["min_delay"].clip(upper=DELAY_CAP_MIN)
    df = df.assign(_d=pd.to_datetime(df["date"], errors="coerce")).sort_values("_d").reset_index(drop=True)
    n = len(df); tr, cal, te = df.iloc[:int(.7*n)], df.iloc[int(.7*n):int(.8*n)], df.iloc[int(.8*n):]
    ytr, ycal, yte = [x["min_delay"].to_numpy().astype(float) for x in (tr, cal, te)]
    Xtr, maps, gm = features(tr, fit=True, y=ytr); Xcal, Xte = features(cal, maps, gm), features(te, maps, gm)

    #stage 1: median + 90% interval
    med = q_model(0.50).fit(Xtr, ytr)
    m_lo, m_hi = q_model(0.05).fit(Xtr, ytr), q_model(0.95).fit(Xtr, ytr)
    Q = conformal_Q(m_lo.predict(Xcal), m_hi.predict(Xcal), ycal, 0.10)
    lo_te = np.clip(m_lo.predict(Xte) - Q, 0, None); hi_te = m_hi.predict(Xte) + Q
    miss = ((yte < lo_te) | (yte > hi_te)).astype(int)

    #stage 2: predict |error| from features. trained on calib so its out of sample
    err_cal = np.abs(ycal - med.predict(Xcal))
    u = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=48, max_depth=7,
                      min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=0.5, random_state=42, verbose=-1).fit(Xcal, err_cal)
    u_te = u.predict(Xte)
    err_te = np.abs(yte - med.predict(Xte))

    r2 = r2_score(err_te, u_te)
    rho = spearmanr(u_te, err_te).statistic
    auc = roc_auc_score(miss, u_te)
    print("=== RQ3: Is the model's own uncertainty predictable? (future fold) ===\n")
    print(f"  Predicting |error| from incident features:")
    print(f"    R^2                = {r2:.3f}")
    print(f"    Spearman corr      = {rho:.3f}   (predicted vs actual error rank)")
    print(f"  Ranking out-of-interval misses by predicted uncertainty:")
    print(f"    AUC                = {auc:.3f}   (0.5 = no skill)")

    #split test into low/med/high predicted uncertainty
    q1, q2 = np.quantile(u_te, [1/3, 2/3])
    band = np.where(u_te <= q1, "1 low", np.where(u_te <= q2, "2 medium", "3 high"))
    tab = pd.DataFrame({"band": band, "actual_error": err_te,
                        "interval_width": hi_te - lo_te, "inside": 1 - miss})
    g = tab.groupby("band").agg(n=("actual_error", "size"), mean_actual_error=("actual_error", "mean"),
                                mean_interval_width=("interval_width", "mean"), coverage=("inside", "mean")).round(2)
    print("\n  By predicted-uncertainty tercile:\n" + g.to_string())
    g.to_csv(OUT / "uncertainty_prediction.csv")
    print(f"\n  Saved -> {OUT/'uncertainty_prediction.csv'}")


if __name__ == "__main__":
    main()
