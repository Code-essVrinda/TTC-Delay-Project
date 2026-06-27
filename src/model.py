"""
the delay range model. predicts a range that is right about 9 times out of 10.
has the features, the quantile models, the conformal step, and the estimator class.
give it an incident (date, time, station, line, bound, code) -> low, mid, high minutes.
no min_gap (it leaks). delays capped at 60.
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

CLEANED = Path("data/processed/cleaned.csv")
MODEL_PATH = Path("models/delay_range.joblib")
RESULTS = Path("results")
DELAY_CAP_MIN = 60   #over an hour = major disruption, exact number doesnt matter
RUSH = set(range(7, 10)) | set(range(16, 19))
RATE_COLS = ["station", "line", "bound", "code"]


def features(df, maps=None, gmean=None, fit=False, y=None):
    """
    build the feature table.
    if fit=True learn the avg-delay maps from train and return them too.
    all features known at incident time so no leakage.
    """
    d = pd.to_datetime(df["date"], errors="coerce")
    hour = (df["time"].astype(str).str.split(":").str[0]
            .replace("", "0").astype(float).fillna(0).astype(int).clip(0, 23))
    X = pd.DataFrame(index=df.index)
    X["hour"] = hour
    X["dow"] = d.dt.dayofweek.fillna(0).astype(int)
    X["month"] = d.dt.month.fillna(1).astype(int)
    X["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    X["is_rush"] = hour.isin(RUSH).astype(int)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    if fit:
        #avg delay per category, from train only. smoothed a bit
        maps, gmean = {}, float(np.mean(y))
        for c in RATE_COLS:
            g = pd.DataFrame({"k": df[c].astype(str).to_numpy(), "y": y}).groupby("k")["y"]
            s, nn = g.sum(), g.count()
            maps[c] = ((s + 15 * gmean) / (nn + 15)).to_dict()
        #station x hour combo
        sh = pd.DataFrame({"k": list(zip(df["station"].astype(str), hour)), "y": y}).groupby("k")["y"]
        s, nn = sh.sum(), sh.count()
        maps["station_hour"] = ((s + 20 * gmean) / (nn + 20)).to_dict()
    for c in RATE_COLS:
        X[f"{c}_mean"] = df[c].astype(str).map(maps[c]).fillna(gmean)
    keys = list(zip(df["station"].astype(str), hour))
    X["station_hour_mean"] = [maps["station_hour"].get(k, gmean) for k in keys]
    return (X, maps, gmean) if fit else X


def q_model(alpha):
    #lightgbm that predicts one quantile
    return LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=500,
                         learning_rate=0.03, num_leaves=48, max_depth=7,
                         min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                         reg_lambda=0.5, random_state=42, verbose=-1)


def conformal_Q(lo_cal, hi_cal, ycal, alpha):
    #widens the interval so coverage hits 1-alpha
    E = np.maximum(lo_cal - ycal, ycal - hi_cal)
    k = min(1.0, np.ceil((len(ycal) + 1) * (1 - alpha)) / len(ycal))
    return float(np.quantile(E, k, method="higher"))


class DelayRangeEstimator:
    #90% range = right 9 out of 10
    def __init__(self, lo_q=0.05, hi_q=0.95):
        self.lo_q, self.hi_q = lo_q, hi_q
        self.alpha = lo_q + (1 - hi_q)
        self.m_lo = self.m_hi = self.m_md = None
        self.Q = None
        self.maps = self.gmean = None

    def fit(self, train_df):
        y = train_df["min_delay"].to_numpy().astype(float)
        X, self.maps, self.gmean = features(train_df, fit=True, y=y)
        self.m_lo = q_model(self.lo_q).fit(X, y)
        self.m_hi = q_model(self.hi_q).fit(X, y)
        self.m_md = q_model(0.50).fit(X, y)
        return self

    def conformalize(self, calib_df):
        #fix coverage using a held out slice
        y = calib_df["min_delay"].to_numpy().astype(float)
        X = features(calib_df, self.maps, self.gmean)
        self.Q = conformal_Q(self.m_lo.predict(X), self.m_hi.predict(X), y, self.alpha)
        return self

    def predict_range(self, df):
        X = features(df, self.maps, self.gmean)
        lo = np.clip(self.m_lo.predict(X) - self.Q, 0, None)
        hi = self.m_hi.predict(X) + self.Q
        mid = np.clip(self.m_md.predict(X), lo, hi)
        return np.round(lo).astype(int), np.round(mid).astype(int), np.round(hi).astype(int)

    def save(self, path=MODEL_PATH):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path); return path

    @staticmethod
    def load(path=MODEL_PATH):
        return joblib.load(path)


def _interval_score(y, lo, hi, alpha):
    #winkler score. punishes wide intervals and misses. lower is better
    return float(np.mean((hi - lo)
                         + (2 / alpha) * np.maximum(0, lo - y)
                         + (2 / alpha) * np.maximum(0, y - hi)))


def main():
    df = pd.read_csv(CLEANED)
    df = df[df["min_delay"] > 0].copy()
    df["min_delay"] = df["min_delay"].clip(upper=DELAY_CAP_MIN)
    df = df.assign(_d=pd.to_datetime(df["date"], errors="coerce")).sort_values("_d").reset_index(drop=True)
    n = len(df)
    tr, cal, te = df.iloc[:int(.7*n)], df.iloc[int(.7*n):int(.8*n)], df.iloc[int(.8*n):]

    est = DelayRangeEstimator().fit(tr).conformalize(cal)
    saved = est.save()

    #check on the future test fold
    yte = te["min_delay"].to_numpy().astype(float)
    lo, mid, hi = est.predict_range(te)
    right = np.mean((yte >= lo) & (yte <= hi))
    width = np.median(hi - lo)
    IS = _interval_score(yte, lo, hi, est.alpha)

    #climatology = one fixed range for everyone
    ytr = tr["min_delay"].to_numpy().astype(float)
    ycal = cal["min_delay"].to_numpy().astype(float)
    g_lo, g_hi = np.quantile(ytr, est.lo_q), np.quantile(ytr, est.hi_q)
    Qg = conformal_Q(np.full(len(ycal), g_lo), np.full(len(ycal), g_hi), ycal, est.alpha)
    glo, ghi = max(0, g_lo - Qg), g_hi + Qg
    IS_g = _interval_score(yte, np.full(len(yte), glo), np.full(len(yte), ghi), est.alpha)

    card = pd.DataFrame([
        {"model": "delay-range (per-incident)", "right_out_of_10": round(right * 10, 1),
         "median_width_min": round(float(width), 1), "interval_score": round(IS, 2)},
        {"model": "climatology (one fixed range)", "right_out_of_10": round(float(np.mean((yte >= glo) & (yte <= ghi))) * 10, 1),
         "median_width_min": round(ghi - glo, 1), "interval_score": round(IS_g, 2)},
    ])
    print("=== Locked model — honest scorecard on unseen future delays ===")
    print(card.to_string(index=False))
    print(f"(lower interval_score is better; model beats climatology by {100*(1-IS/IS_g):.0f}%)")
    RESULTS.mkdir(parents=True, exist_ok=True)
    card.to_csv(RESULTS / "range_scorecard.csv", index=False)
    print(f"saved model -> {saved}  |  scorecard -> {RESULTS/'range_scorecard.csv'}")

    #reload and show a few predictions
    est2 = DelayRangeEstimator.load()
    lo, mid, hi = est2.predict_range(te.head(6))
    s = te.head(6)[["station", "code", "min_delay"]].copy()
    s["most_likely"] = [f"{m} min" for m in mid]
    s["range_9of10"] = [f"{a}-{b} min" for a, b in zip(lo, hi)]
    s["actual"] = [f"{int(x)} min" for x in s["min_delay"]]
    print("\nDemo (reloaded model):")
    print(s[["station", "code", "most_likely", "range_9of10", "actual"]].to_string(index=False))


if __name__ == "__main__":
    main()
