"""Convergence check of the 2,000-iteration SA fits on reshuffled sequences.

For each DANA user active >= 8 days (the 40 users of the reshuffling analysis), on the SAME
reshuffled sequence used in notebook 01 (week_reshuffle_iets(seed=42), index 0):
  * theta_stored : parameters in circadian_cycles_all_reshuffle_results.csv (SA 2,000 it.)
  * theta_2000b  : a new, independent SA run with 2,000 iterations  -> run-to-run noise
  * theta_5000   : a new SA run with 5,000 iterations (as for the original data)
Each fit is scored with A_fresh (mean area statistic over R fresh CNHPP realizations).
If A_fresh(theta_5000) is not systematically below A_fresh(theta_stored) beyond the
2,000-vs-2,000 noise, the 2,000-iteration fits had converged and the Monte Carlo p-values
computed from them stand.

Usage: python 08_sa_convergence_check.py [R] [n_jobs]   (defaults 100, all cores)
Writes ../../results/cnhpp/sa_convergence_check_dana.csv (saved after every user)
"""
import contextlib
import io
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import cnhpp                                                    # noqa: E402
from tgdata import load_messages                                # noqa: E402
import importlib.util as _u                                     # noqa: E402
_spec = _u.spec_from_file_location("fair", os.path.join(HERE, "07_reshuffle_fair_comparison.py"))
fair = _u.module_from_spec(_spec); _spec.loader.exec_module(fair)

OUT = os.path.join(HERE, "..", "..", "results", "cnhpp", "sa_convergence_check_dana.csv")
SEED = 7


def fit_params(events_h, n_iter):
    with contextlib.redirect_stdout(io.StringIO()):
        cfg, A_best = cnhpp.simulated_annealing(events_h, n_iter=n_iter)
    rho_a, Nw, pd_t, pw_t = cnhpp.compute_model_params(events_h, np.concatenate([[False], cfg]))
    return (rho_a, Nw, pd_t, pw_t, cnhpp.get_pNa(cfg)), A_best


def job(args):
    uid, dates, row_r, R, seed = args
    random.seed(seed); np.random.seed(seed % 2**32)
    _, shuf_h = fair.user_sequences(dates)
    iets, T, N = np.diff(shuf_h), shuf_h[-1], len(shuf_h)
    th_b, Ab_b = fit_params(shuf_h, 2000)
    th_5, Ab_5 = fit_params(shuf_h, 5000)
    A_s, se_s = fair.a_fresh(iets, fair.params_of(row_r), T, N, R)
    A_b, se_b = fair.a_fresh(iets, th_b, T, N, R)
    A_5, se_5 = fair.a_fresh(iets, th_5, T, N, R)
    return dict(user_id=uid, n_events=N, p_resh=float(row_r["p_value_reshuffle"]),
                A_fresh_stored=A_s, A_fresh_2000b=A_b, A_fresh_5000=A_5,
                se_stored=se_s, se_2000b=se_b, se_5000=se_5,
                A_best_stored=float(row_r["A_best"]), A_best_2000b=Ab_b, A_best_5000=Ab_5)


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    df = load_messages("dana")
    r = pd.read_csv(os.path.join(fair.RES, "circadian_cycles_all_reshuffle_results.csv")).set_index("user_id")
    seeds = iter(np.random.SeedSequence(SEED).generate_state(500))
    jobs = []
    for uid in r.index:
        dates = df.loc[df["sender_id"] == uid, "date"].dropna().sort_values()
        if (dates.max() - dates.min()).days >= 8:
            jobs.append((int(uid), dates, r.loc[uid], R, int(next(seeds))))
    jobs.sort(key=lambda j: -len(j[1]))            # longest first for load balance
    rows = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for f in as_completed([ex.submit(job, j) for j in jobs]):
            rows.append(f.result())
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"{len(rows)}/{len(jobs)} done", flush=True)
    out = pd.DataFrame(rows)
    out["gain_5000"] = out.A_fresh_stored - out.A_fresh_5000        # > 0: 5000 fits better
    out["noise_2000"] = (out.A_fresh_stored - out.A_fresh_2000b).abs()
    out.to_csv(OUT, index=False)
    print(f"median A_fresh  stored {out.A_fresh_stored.median():.3f} | 2000b {out.A_fresh_2000b.median():.3f} "
          f"| 5000 {out.A_fresh_5000.median():.3f}")
    print(f"median gain 5000 vs stored {out.gain_5000.median():+.3f}  vs median |2000-2000| noise {out.noise_2000.median():.3f}")
    print(f"users where 5000 beats stored by more than 2x the 2000-vs-2000 noise: "
          f"{(out.gain_5000 > 2 * out.noise_2000).sum()}/{len(out)}")


if __name__ == "__main__":
    main()
