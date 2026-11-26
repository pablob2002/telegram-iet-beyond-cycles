"""Is the CNHPP rejected because of power (more messages) or because of temporal concentration?

Per user (same users and windows as notebooks 01-03):
  n_events      number of messages (statistical power of the Monte Carlo test grows with it)
  span_days     time between first and last message
  active_weeks  number of distinct weeks with at least one message
  conc3         fraction of the user's messages in their 3 busiest weeks (temporal concentration)
Responses:
  rejected      CNHPP rejected on the original data (Monte Carlo p < 0.05, M = 100)
  improvement   relative drop of A_fresh after week reshuffling (from 07_reshuffle_fair_comparison.py)

Analyses: rejection rate by n_events band and dataset; logistic regressions
rejected ~ DANA  vs  rejected ~ DANA + log n + conc3 (+ log span); Spearman correlations of the
reshuffling improvement with conc3 and n_events.

Writes ../../results/cnhpp/rejection_covariates.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from tgdata import load_messages  # noqa: E402

RES = os.path.join(ROOT, "results", "cnhpp")
DATASETS = [("DANA", "dana", (None, None), "circadian_cycles_all"),
            ("RENFE 2024", "vagarenfe", ("2024-01-01", "2024-12-31"), "circadian_cycles_vagarenfe_2024"),
            ("RENFE 2026", "vagarenfe", ("2026-01-15", None), "circadian_cycles_vagarenfe_2026")]


def covariates(dates):
    wk = dates.dt.tz_convert("Europe/Madrid").dt.tz_localize(None).dt.to_period("W")
    counts = wk.value_counts()
    return dict(n_events=len(dates), span_days=(dates.max() - dates.min()).total_seconds() / 86400,
                active_weeks=len(counts), conc3=counts.nlargest(3).sum() / len(dates))


def main():
    msgs = {c: load_messages(c) for c in ("dana", "vagarenfe")}
    fair = pd.read_csv(os.path.join(RES, "reshuffle_fair_comparison.csv"))
    rows = []
    for name, comm, (dmin, dmax), stem in DATASETS:
        df = msgs[comm]
        if dmin:
            df = df[df["date"] >= dmin]
        if dmax:
            df = df[df["date"] <= dmax]
        res = pd.read_csv(os.path.join(RES, f"{stem}_results.csv"))
        for _, r in res.iterrows():
            dates = df.loc[df["sender_id"] == r.user_id, "date"].sort_values()
            rows.append(dict(dataset=name, user_id=int(r.user_id), p_value=r.p_value, **covariates(dates)))
    d = pd.DataFrame(rows)
    d["rejected"] = (d.p_value < 0.05).astype(int)
    d["DANA"] = (d.dataset == "DANA").astype(int)
    d["log_n"] = np.log10(d.n_events)
    d["log_span"] = np.log10(d.span_days)
    d = d.merge(fair[["dataset", "user_id", "rel_change_fresh", "min8days"]], on=["dataset", "user_id"], how="left")
    d["improvement"] = -d.rel_change_fresh
    d.to_csv(os.path.join(RES, "rejection_covariates.csv"), index=False)

    pd.set_option("display.width", 200)
    print("== Covariates by dataset (medians) and rejection rate ==")
    print(d.groupby("dataset", sort=False).agg(users=("user_id", "size"), n_events=("n_events", "median"),
          span_days=("span_days", "median"), active_weeks=("active_weeks", "median"),
          conc3=("conc3", "median"), rejected=("rejected", "mean")).round(2).to_string())

    print("\n== Rejection rate by number of messages ==")
    d["n_band"] = pd.cut(d.n_events, [40, 100, 200, 10_000], labels=["41-100", "101-200", ">200"])
    tab = d.groupby(["n_band", "dataset"], observed=True).rejected.agg(["sum", "size"])
    print(tab.assign(rate=lambda x: (x["sum"] / x["size"]).round(2)).to_string())
    big = d[d.n_events > 100]
    print("\nUsers with > 100 messages (the RENFE threshold): rejection rate "
          + ", ".join(f"{k} {g.rejected.sum()}/{len(g)}" for k, g in big.groupby("dataset", sort=False)))

    print("\n== Logistic regressions (rejected) ==")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in ["rejected ~ DANA", "rejected ~ log_n + conc3", "rejected ~ DANA + log_n + conc3",
                  "rejected ~ DANA + log_n + conc3 + log_span"]:
            m = smf.logit(f, data=d).fit(disp=0)
            coefs = ", ".join(f"{k}={v:+.2f} (p={m.pvalues[k]:.3f})" for k, v in m.params.items() if k != "Intercept")
            print(f"{f:45s} AIC={m.aic:6.1f}  {coefs}")
        dd = d[d.DANA == 1]
        m = smf.logit("rejected ~ log_n + conc3", data=dd).fit(disp=0)
        print(f"{'DANA only: rejected ~ log_n + conc3':45s} AIC={m.aic:6.1f}  "
              + ", ".join(f"{k}={v:+.2f} (p={m.pvalues[k]:.3f})" for k, v in m.params.items() if k != "Intercept"))

    print("\n== Reshuffling improvement (relative drop of A_fresh) ==")
    e = d.dropna(subset=["improvement"])
    for lab, g in [("all users", e), ("DANA >= 8 days + RENFE", e[(e.DANA == 0) | (e.min8days == True)]),
                   ("DANA only", e[e.DANA == 1]), ("RENFE only", e[e.DANA == 0])]:
        r1, p1 = spearmanr(g.improvement, g.conc3)
        r2, p2 = spearmanr(g.improvement, g.n_events)
        print(f"{lab:26s} n={len(g):3d}  rho(improvement, conc3)={r1:+.2f} (p={p1:.1e})   "
              f"rho(improvement, n_events)={r2:+.2f} (p={p2:.1e})")
    m = smf.ols("improvement ~ conc3 + log_n + DANA", data=e[(e.DANA == 0) | (e.min8days == True)]).fit()
    print("OLS improvement ~ conc3 + log_n + DANA (users >= 8 days): "
          + ", ".join(f"{k}={v:+.2f} (p={m.pvalues[k]:.3f})" for k, v in m.params.items() if k != "Intercept"))


if __name__ == "__main__":
    main()
