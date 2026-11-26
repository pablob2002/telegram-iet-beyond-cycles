"""Fair comparison of the CNHPP fit on original vs week-reshuffled sequences.

Why: A_best (the SA objective) is the minimum of many noisy evaluations, so it is
optimistically biased and the bias grows with the number of SA iterations. The original
fits used 5,000 iterations and the reshuffled ones 2,000, so comparing A_best is unfair.
Here both fits are scored the same way: A_fresh = mean area statistic over R fresh CNHPP
realizations with the stored parameters (results/cnhpp/*_results.csv and *_reshuffle_results.csv).

Sequences are rebuilt exactly as in notebooks 01-03: same users, same windows, same
reshuffled sequence (week_reshuffle_iets(seed=42), permutation index 0).

Also tabulates, per user, the Monte Carlo decision before and after reshuffling
(rescued = rejected -> accepted; newly rejected = accepted -> rejected).

Usage: python 07_reshuffle_fair_comparison.py [R] [n_jobs]     (defaults 100, all cores)
Writes ../../results/cnhpp/reshuffle_fair_comparison.csv
"""
import ast
import io
import contextlib
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from tgdata import load_messages  # noqa: E402
import cnhpp                      # noqa: E402

RES = os.path.join(ROOT, "results", "cnhpp")
SEED = 2026
DATASETS = [  # name, community, (dmin, dmax), csv stem
    ("DANA", "dana", (None, None), "circadian_cycles_all"),
    ("RENFE 2024", "vagarenfe", ("2024-01-01", "2024-12-31"), "circadian_cycles_vagarenfe_2024"),
    ("RENFE 2026", "vagarenfe", ("2026-01-15", None), "circadian_cycles_vagarenfe_2026"),
]


def parse_array(s):
    s = str(s).strip().replace("\n", " ")
    if "," not in s:
        return np.array([float(x) for x in s.replace("[", "").replace("]", "").split()])
    return np.array(ast.literal_eval(s))


def params_of(row):
    return (float(row["rho_a"]), float(row["Nw"]), parse_array(row["pd_t"]),
            parse_array(row["pw_t"]), parse_array(row["p_Na"]))


def a_fresh(iets_h, params, total_h, n_events, R):
    vals = []
    for _ in range(R):
        sim = cnhpp.simulate_cascading_nhpp(*params, total_hours=total_h, n_samples=n_events)
        if len(sim) > 1:
            vals.append(cnhpp.area_test_statistic(iets_h, np.diff(sim)))
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))


def user_sequences(dates):
    """Original event times (h) and reshuffled event times (h), as in notebooks 01-03."""
    ev_h = ((dates - dates.iloc[0]).dt.total_seconds() / 3600).to_numpy()
    with contextlib.redirect_stdout(io.StringIO()):
        _, shuf_list = cnhpp.week_reshuffle_iets(dates, n_shuffles=200, seed=42)
    s = np.asarray(shuf_list[0], float)
    s = s[s > 0]
    shuf_h = np.concatenate([[0.0], np.cumsum(s / 3600.0)])
    return ev_h, shuf_h


def job(args):
    name, uid, dates, p_orig, p_resh, row_o, row_r, R, seed = args
    random.seed(seed); np.random.seed(seed % 2**32)
    ev_h, shuf_h = user_sequences(dates)
    A_o, se_o = a_fresh(np.diff(ev_h), params_of(row_o), ev_h[-1] - ev_h[0], len(ev_h), R)
    A_r, se_r = a_fresh(np.diff(shuf_h), params_of(row_r), shuf_h[-1], len(shuf_h), R)
    span_days = (dates.max() - dates.min()).days
    return dict(dataset=name, user_id=uid, n_events=len(ev_h), span_days=span_days,
                min8days=span_days >= 8, p_orig=p_orig, p_resh=p_resh,
                A_best_orig=row_o["A_best"], A_best_resh=row_r["A_best"],
                A_fresh_orig=A_o, A_fresh_orig_se=se_o, A_fresh_resh=A_r, A_fresh_resh_se=se_r)


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    msgs = {c: load_messages(c) for c in ("dana", "vagarenfe")}
    seeds = iter(np.random.SeedSequence(SEED).generate_state(1000))
    jobs = []
    for name, comm, (dmin, dmax), stem in DATASETS:
        df = msgs[comm]
        if dmin:
            df = df[df["date"] >= dmin]
        if dmax:
            df = df[df["date"] <= dmax]
        o = pd.read_csv(os.path.join(RES, f"{stem}_results.csv")).set_index("user_id")
        r = pd.read_csv(os.path.join(RES, f"{stem}_reshuffle_results.csv")).set_index("user_id")
        for uid in r.index:
            dates = df.loc[df["sender_id"] == uid, "date"].dropna().sort_values()
            jobs.append((name, int(uid), dates, float(o.loc[uid, "p_value"]), float(r.loc[uid, "p_value_reshuffle"]),
                         o.loc[uid], r.loc[uid], R, int(next(seeds))))
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        out = pd.DataFrame(list(ex.map(job, jobs)))
    out["rel_change_fresh"] = out.A_fresh_resh / out.A_fresh_orig - 1
    out["rel_change_best"] = out.A_best_resh / out.A_best_orig - 1
    out.to_csv(os.path.join(RES, "reshuffle_fair_comparison.csv"), index=False)

    rows = []
    groups = [(n, g) for n, g in out.groupby("dataset", sort=False)]
    groups.insert(1, ("DANA (>=8 days)", out[(out.dataset == "DANA") & out.min8days]))
    for name, g in groups:
        rej_o, rej_r = g.p_orig < 0.05, g.p_resh < 0.05
        rows.append(dict(
            dataset=name, users=len(g),
            A_best_change=f"{g.rel_change_best.median():+.0%}",
            A_fresh_orig=round(g.A_fresh_orig.median(), 3), A_fresh_resh=round(g.A_fresh_resh.median(), 3),
            A_fresh_change=f"{g.rel_change_fresh.median():+.0%}",
            wilcoxon_p=f"{wilcoxon(g.A_fresh_orig, g.A_fresh_resh, alternative='greater').pvalue:.1e}",
            rejected=f"{rej_o.sum()} -> {rej_r.sum()}",
            rescued=int((rej_o & ~rej_r).sum()), newly_rejected=int((~rej_o & rej_r).sum())))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
