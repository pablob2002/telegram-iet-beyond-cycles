"""Table I: global inter-event-time fits for the three datasets.

For each dataset this script computes, on the same IETs:
  * the truncated power law (TPL) MLE fit (alpha, lambda, xmin) with `powerlaw`;
  * the KS distance D between the empirical tail (tau >= xmin) and the fitted TPL;
  * 95% confidence intervals for alpha and lambda by nonparametric bootstrap
    (resample IETs with replacement, refit, re-estimating xmin each time);
  * the semiparametric bootstrap goodness-of-fit p-value of Clauset, Shalizi &
    Newman (2009): each synthetic sample draws its body (tau < xmin) from the
    empirical body and its tail from the fitted TPL, in the empirical proportion;
    the model is refitted (xmin re-estimated) and p = fraction with D_b >= D_emp.

Windows: DANA up to 2025-01-01; RENFE 2024 = calendar year 2024; RENFE 2026 from
2026-01-15. These are the windows of waiting_time_fit_combined.py.

Usage:  python table1_global_fits.py [B] [n_jobs]      (defaults: B=500, all cores)
Writes: ../../results/iet/table1_global_fits.csv
"""
import os, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import powerlaw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from tgdata import load_messages  # noqa: E402

DISCRETE = True   # timestamps have 1 s resolution; same choice as the channel/user fits
SEED = 2026
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "iet", "table1_global_fits.csv")


def global_iets(df, dmin, dmax):
    d = pd.to_datetime(df["date"], errors="coerce").dropna()
    if d.dt.tz is not None:
        d = d.dt.tz_localize(None)
    d = d.sort_values()
    if dmin:
        d = d[d >= dmin]
    if dmax:
        d = d[d < dmax]
    w = d.diff().dt.total_seconds().dropna().values
    return w[(w > 0) & np.isfinite(w)]


def fit_tpl(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = powerlaw.Fit(x, discrete=DISCRETE, verbose=False)
        t = f.truncated_power_law
        return float(t.alpha), float(t.Lambda), float(t.xmin), float(t.D), t


def _boot_ci(args):
    x, seed = args
    rng = np.random.default_rng(seed)
    a, lam, *_ = fit_tpl(rng.choice(x, size=len(x), replace=True))
    return a, lam


def sample_tpl(n, xmin, alpha, lam, rng):
    """Draw n values from the TPL p(x) ~ x^-alpha exp(-lam x), x >= xmin.

    Exact rejection sampler for the continuous TPL (Pareto proposal, accept with
    prob exp(-lam (x - xmin))). For DISCRETE fits the continuous law is started
    at xmin - 1/2 and rounded to the nearest integer, the approximation of
    Clauset et al. (2009, App. D), accurate here because xmin >> 1 s.
    """
    x0 = xmin - 0.5 if DISCRETE else xmin
    out = np.empty(0)
    while out.size < n:
        m = int(1.3 * (n - out.size)) + 16
        prop = x0 * (1.0 - rng.random(m)) ** (-1.0 / (alpha - 1.0))
        keep = rng.random(m) < np.exp(-lam * (prop - x0))
        out = np.concatenate([out, prop[keep]])
    out = out[:n]
    return np.floor(out + 0.5) if DISCRETE else out


def _boot_gof(args):
    x, xmin, alpha, lam, seed = args
    rng = np.random.default_rng(seed)
    body = x[x < xmin]
    n_tail = rng.binomial(len(x), np.mean(x >= xmin))
    synth = np.concatenate([rng.choice(body, size=len(x) - n_tail, replace=True),
                            sample_tpl(n_tail, xmin, alpha, lam, rng)])
    return fit_tpl(synth)[3]


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    dana, renfe = load_messages("dana"), load_messages("vagarenfe")
    datasets = [("DANA", global_iets(dana, None, "2025-01-01")),
                ("RENFE 2024", global_iets(renfe, "2024-01-01", "2025-01-01")),
                ("RENFE 2026", global_iets(renfe, "2026-01-15", None))]
    ss = np.random.SeedSequence(SEED)
    rows = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for name, x in datasets:
            t0 = time.time()
            alpha, lam, xmin, D, _ = fit_tpl(x)
            seeds = ss.spawn(2 * B)
            ci = np.array(list(ex.map(_boot_ci, [(x, s) for s in seeds[:B]], chunksize=4)))
            Db = np.array(list(ex.map(_boot_gof, [(x, xmin, alpha, lam, s) for s in seeds[B:]], chunksize=4)))
            a_lo, a_hi = np.percentile(ci[:, 0], [2.5, 97.5])
            l_lo, l_hi = np.percentile(ci[:, 1], [2.5, 97.5])
            row = dict(dataset=name, n_iet=len(x), xmin_s=xmin, n_tail=int(np.sum(x >= xmin)),
                       alpha=alpha, alpha_ci_lo=a_lo, alpha_ci_hi=a_hi,
                       lambda_per_s=lam, lambda_ci_lo=l_lo, lambda_ci_hi=l_hi,
                       ks_D=D, gof_p=float(np.mean(Db >= D)), B=B, discrete=DISCRETE)
            rows.append(row)
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)   # saved after each dataset
            print(f"{name}: alpha={alpha:.3f} [{a_lo:.3f}, {a_hi:.3f}]  lambda={lam:.2e} "
                  f"[{l_lo:.2e}, {l_hi:.2e}]  D={D:.4f}  GoF p={row['gof_p']:.3f}  ({time.time()-t0:.0f}s)",
                  flush=True)
    print("written", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
