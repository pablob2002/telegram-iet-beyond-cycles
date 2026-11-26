"""Pilot: Hawkes process with cyclic baseline for six RENFE 2026 users.

Model per user (src/hawkes_cyclic.py), time in hours:
    lambda(t) = mu0 * w(t) * c(t) + n * beta * sum_{t_i<t} exp(-beta (t - t_i))
  w(t): weekly profile (168 local-time hour-of-week slots, Europe/Madrid) of all OTHER users'
        messages in the window, circularly smoothed over +-1 h, mean 1;
  c(t): daily activity of all OTHER users (3-day centred mean), mean 1 over the user's span.
Variant "cycle_only" drops c(t) (to see how much non-stationarity inflates n).

Per user and variant:
  * MLE of (mu0, n, beta); null model n = 0 (inhomogeneous Poisson mu0 w c);
  * LR statistic D = 2 (logL_hawkes - logL_null), p-value by parametric bootstrap under the null
    (the usual chi-square does not apply: beta is not identified when n = 0);
  * time-rescaling KS test (Exp(1)) for Hawkes and for the null;
  * B and M_log from R simulations of the fitted Hawkes / null, against the empirical values
    and the CNHPP medians (results/cnhpp/user_burstiness_memory.csv).

Usage: python 01_hawkes_pilot_renfe2026.py [B_boot] [R_sim] [n_jobs]   (defaults 200 200 all cores)
Writes ../../results/hawkes/pilot_renfe2026.csv and figures/pilot_rescaling_qq.pdf
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import kstest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from tgdata import load_messages  # noqa: E402
import hawkes_cyclic as hk        # noqa: E402

PILOT_USERS = [101779, 100662, 101572, 101101, 100899, 101370]   # pseudonyms; see README
WINDOW_START = "2026-01-15"
TZ = "Europe/Madrid"
SEED = 2026


def b_and_m(t_hours):
    tau = np.diff(t_hours) * 3600.0
    mu, sd = tau.mean(), tau.std()
    B = (sd - mu) / (sd + mu)
    lt = np.log(tau[tau > 0])
    M = np.corrcoef(lt[:-1], lt[1:])[0, 1] if len(lt) > 3 else np.nan
    return B, M


def build_user(df, uid, rng):
    """Event times (h) and hourly baseline grids for one user."""
    loc = df["date"].dt.tz_convert(TZ)
    origin = loc.min().floor("D")                                  # local midnight
    hours_total = int(np.ceil((loc.max() - origin).total_seconds() / 3600)) + 2
    grid = origin + pd.to_timedelta(np.arange(hours_total), unit="h")
    others = loc[df["sender_id"] != uid]
    # weekly profile w (hour-of-week), smoothed, mean 1 over the 168 slots
    how = (others.dt.dayofweek * 24 + others.dt.hour).to_numpy()
    prof = np.bincount(how, minlength=168).astype(float) + 1.0
    prof = (np.roll(prof, 1) + 2 * prof + np.roll(prof, -1)) / 4
    prof /= prof.mean()
    w = prof[(grid.dayofweek * 24 + grid.hour).to_numpy()]
    # daily community activity c
    days = pd.date_range(origin, grid[-1].floor("D"), freq="D")
    daily = others.dt.floor("D").value_counts().reindex(days, fill_value=0).astype(float)
    daily = daily.rolling(3, center=True, min_periods=1).mean() + 0.5
    c = daily.reindex(grid.floor("D")).to_numpy()
    # user events, ties broken uniformly within the second
    ev = loc[df["sender_id"] == uid].sort_values()
    t = ((ev - origin).dt.total_seconds().to_numpy() + rng.random(len(ev))) / 3600.0
    t = np.sort(t)
    span = slice(int(t[0]), int(t[-1]) + 1)
    c = c / c[span].mean()
    return t, w, w * c


def analyse(job):
    uid, t, bases, B_boot, R, seed = job
    rng = np.random.default_rng(seed)
    B_emp, M_emp = b_and_m(t)
    row = dict(user_id=uid, n_events=len(t), B_emp=B_emp, Mlog_emp=M_emp)
    qq = {}
    for name, b in bases.items():
        Cb = hk.cumbase(b)
        null = hk.fit_null(t, b, Cb)
        fit = hk.fit_hawkes(t, b, Cb)
        D = 2 * (fit["loglik"] - null["loglik"])
        # parametric bootstrap of D under the null
        Db = []
        for _ in range(B_boot):
            s = hk.simulate(null["mu0"], 0.0, 1.0, b, t[0], t[-1], int(rng.integers(2**31 - 1)))
            if len(s) < 5:
                continue
            Db.append(2 * (hk.fit_hawkes(s, b, hk.cumbase(b), beta_starts=(60.0, 1.0, 0.1),
                                         n_starts=(0.3,))["loglik"] - hk.fit_null(s, b, hk.cumbase(b))["loglik"]))
        z_h = hk.rescaled_iets(fit["mu0"], fit["n"], fit["beta"], t, b, Cb)
        z_0 = hk.rescaled_iets(null["mu0"], 0.0, 1.0, t, b, Cb)
        # B, M_log under fitted Hawkes and null
        sims_h = [b_and_m(hk.simulate(fit["mu0"], fit["n"], fit["beta"], b, t[0], t[-1],
                                       int(rng.integers(2**31 - 1)))) for _ in range(R)]
        sims_0 = [b_and_m(hk.simulate(null["mu0"], 0.0, 1.0, b, t[0], t[-1],
                                       int(rng.integers(2**31 - 1)))) for _ in range(R)]
        sh, s0 = np.array(sims_h), np.array(sims_0)
        row.update({
            f"{name}_n": fit["n"], f"{name}_tau_min": 60.0 / fit["beta"], f"{name}_mu0": fit["mu0"],
            f"{name}_D": D, f"{name}_p_selfexc": float(np.mean(np.array(Db) >= D)),
            f"{name}_KS_hawkes_p": kstest(z_h, "expon").pvalue, f"{name}_KS_null_p": kstest(z_0, "expon").pvalue,
            f"{name}_KS_hawkes_D": kstest(z_h, "expon").statistic, f"{name}_KS_null_D": kstest(z_0, "expon").statistic,
            f"{name}_B_hawkes_med": np.nanmedian(sh[:, 0]), f"{name}_p_B_hawkes": np.nanmean(sh[:, 0] >= B_emp),
            f"{name}_Mlog_hawkes_med": np.nanmedian(sh[:, 1]), f"{name}_p_Mlog_hawkes": np.nanmean(sh[:, 1] >= M_emp),
            f"{name}_B_null_med": np.nanmedian(s0[:, 0]), f"{name}_Mlog_null_med": np.nanmedian(s0[:, 1]),
        })
        qq[name] = (np.sort(z_h), np.sort(z_0))
    return row, qq


def main():
    B_boot = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    msgs = load_messages("vagarenfe")
    df = msgs[msgs["date"] >= WINDOW_START].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    jobs = []
    for uid in PILOT_USERS:
        t, w, wc = build_user(df, uid, rng)
        jobs.append((uid, t, {"full": wc, "cycle_only": w}, B_boot, R, int(rng.integers(2**31 - 1))))
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        results = list(ex.map(analyse, jobs))
    out = pd.DataFrame([r for r, _ in results])
    cn = pd.read_csv(os.path.join(ROOT, "results", "cnhpp", "user_burstiness_memory.csv"))
    cn = cn[cn.dataset == "RENFE 2026"][["user_id", "B_cnhpp_median", "M_log_cnhpp_median"]]
    out = out.merge(cn, on="user_id", how="left")
    os.makedirs(os.path.join(ROOT, "results", "hawkes"), exist_ok=True)
    out.to_csv(os.path.join(ROOT, "results", "hawkes", "pilot_renfe2026.csv"), index=False)

    pd.set_option("display.width", 220)
    show = out[["user_id", "n_events", "full_n", "full_tau_min", "full_p_selfexc", "cycle_only_n",
                "full_KS_null_D", "full_KS_hawkes_D", "full_KS_hawkes_p",
                "B_emp", "B_cnhpp_median", "full_B_hawkes_med", "full_p_B_hawkes",
                "Mlog_emp", "M_log_cnhpp_median", "full_Mlog_hawkes_med", "full_p_Mlog_hawkes"]]
    print(show.round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
    for ax, (row, qq) in zip(axes.flat, results):
        zh, z0 = qq["full"]
        q = lambda z: -np.log(1 - (np.arange(1, len(z) + 1) - 0.5) / len(z))
        ax.loglog(q(z0), z0, ".", ms=3, c="#999999", label="Cyclic Poisson (n=0)")
        ax.loglog(q(zh), zh, ".", ms=3, c="#D55E00", label="Hawkes")
        lim = [1e-4, 20]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlim(lim); ax.set_ylim(1e-6, 50)
        ax.set_title(f"user {row['user_id']}  (N={row['n_events']}, n={row['full_n']:.2f})", fontsize=9)
        ax.set_xlabel("Exp(1) quantile"); ax.set_ylabel("rescaled IET")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "pilot_rescaling_qq.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
