"""Per-user burstiness B and memory coefficient M, compared with the fitted CNHPP.

Users and event sequences are exactly those of the CNHPP notebooks 01-03:
  DANA: > 40 messages, whole export;  RENFE 2024: > 100 messages, 2024-01-01..2024-12-31;
  RENFE 2026: > 100 messages, from 2026-01-15.  IETs are all consecutive differences (s).

Statistics (Goh & Barabási, EPL 81, 48002, 2008):
  B   = (sigma - mu) / (sigma + mu)                      burstiness, -1 regular .. 0 Poisson .. 1 bursty
  B_n = finite-size corrected burstiness (Kim & Jo, PRE 94, 032311, 2016)
  M   = Pearson correlation between consecutive IETs (tau_i, tau_{i+1})   memory coefficient
  M_log = same on log IETs (zero IETs dropped); robust to a few very long gaps

Two reference ensembles per user:
  * shuffle: the user's IETs in random order (same distribution, no temporal order);
    p_M_shuffle = P(M_shuffle >= M_emp)  -> is there any memory at all?
  * CNHPP: R realizations of the cascading NHPP with the user's fitted parameters
    (results/cnhpp/*_results.csv), same number of events and time span as the data;
    p_M_cnhpp = P(M_cnhpp >= M_emp), p_B_cnhpp = P(B_cnhpp >= B_emp)
    -> does the cycle-driven model reproduce the observed memory / burstiness?

Usage:  python 06_user_burstiness_memory.py [R] [n_shuffle] [n_jobs]   (defaults 200 1000 all cores)
Writes: ../../results/cnhpp/user_burstiness_memory.csv and figures/user_burstiness_memory.pdf
"""
import ast
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from tgdata import load_messages  # noqa: E402

RES = os.path.join(HERE, "..", "..", "results", "cnhpp")
SEED = 2026
DATASETS = [  # name, community, date filter, min messages, result csv
    ("DANA", "dana", (None, None), 40, "circadian_cycles_all_results.csv"),
    ("RENFE 2024", "vagarenfe", ("2024-01-01", "2024-12-31"), 100, "circadian_cycles_vagarenfe_2024_results.csv"),
    ("RENFE 2026", "vagarenfe", ("2026-01-15", None), 100, "circadian_cycles_vagarenfe_2026_results.csv"),
]


# --- statistics -----------------------------------------------------------
def burstiness(t):
    mu, sd = np.mean(t), np.std(t)
    return (sd - mu) / (sd + mu) if sd + mu > 0 else np.nan


def burstiness_n(t):
    """Kim & Jo (2016) finite-size corrected burstiness, n = number of IETs."""
    n = len(t)
    mu, sd = np.mean(t), np.std(t)
    if mu == 0 or n < 2:
        return np.nan
    r = sd / mu
    return (np.sqrt(n + 1) * r - np.sqrt(n - 1)) / ((np.sqrt(n + 1) - 2) * r + np.sqrt(n - 1))


def memory_log(t):
    """Robust variant: memory coefficient of log IETs (zero IETs dropped)."""
    t = np.asarray(t, float)
    t = np.log(t[t > 0])
    return memory(t) if len(t) > 3 else np.nan


def memory(t):
    a, b = t[:-1], t[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# --- CNHPP simulator: identical to simulate_cascading_nhpp in 01-03_CNHPP_*.ipynb ----
def simulate_cascading_nhpp(rho_a, Nw, pd_t, pw_t, p_Na, total_hours, n_samples=None):
    pd_t_arr = np.asarray(pd_t, dtype=float); pd_t_arr = pd_t_arr / pd_t_arr.sum()
    pw_t_arr = np.asarray(pw_t, dtype=float); pw_t_arr = pw_t_arr / pw_t_arr.sum()
    p_Na_norm = np.asarray(p_Na, dtype=float); p_Na_norm = p_Na_norm / p_Na_norm.sum()
    mean_Na = np.dot(np.arange(len(p_Na_norm)), p_Na_norm)
    sim_hours = total_hours
    if n_samples is not None and Nw > 0:
        expected_events = Nw * (total_hours / (7 * 24)) * (1 + mean_Na)
        if expected_events < n_samples:
            sim_hours = total_hours * (n_samples / max(1, expected_events)) * 1.5
    n_weeks = sim_hours / (7 * 24)
    joint = np.outer(pw_t_arr, pd_t_arr).flatten(); joint /= joint.sum()
    n_bursts = np.random.poisson(max(1.0, Nw * n_weeks))
    week_nums = np.random.uniform(0, n_weeks, n_bursts)
    slots = np.random.choice(168, size=n_bursts, p=joint)
    burst_times = (np.floor(week_nums) * 7 * 24 + (slots // 24) * 24 + (slots % 24)
                   + np.random.uniform(0, 1, n_bursts))
    burst_times = np.sort(burst_times)
    burst_times = burst_times[burst_times < sim_hours]
    if rho_a <= 0:
        rho_a = 1.0
    events = []
    for bt in burst_times:
        events.append(bt)
        Na = np.random.choice(len(p_Na_norm), p=p_Na_norm)
        t = bt
        for _ in range(Na):
            t += np.random.exponential(1.0 / rho_a)
            events.append(t)
    result = np.sort(np.array(events))
    return result[:n_samples] if n_samples is not None else result


def parse_array(s):
    s = str(s).strip().replace("\n", " ")
    if "," not in s:
        return np.array([float(x) for x in s.replace("[", "").replace("]", "").split()])
    return np.array(ast.literal_eval(s))


# --- per-user job ----------------------------------------------------------
def analyse_user(job):
    name, uid, ev_s, params, R, n_shuffle, seed = job
    rng = np.random.default_rng(seed)
    np.random.seed(int(rng.integers(2**31 - 1)))       # the simulator uses the global RNG
    tau = np.diff(ev_s)                                  # seconds, all consecutive IETs
    B, Bn, M, ML = burstiness(tau), burstiness_n(tau), memory(tau), memory_log(tau)
    perms = [rng.permutation(tau) for _ in range(n_shuffle)]
    M_sh = np.array([memory(x) for x in perms]); ML_sh = np.array([memory_log(x) for x in perms])
    total_h = (ev_s[-1] - ev_s[0]) / 3600.0
    M_c, B_c, ML_c = [], [], []
    for _ in range(R):
        sim = simulate_cascading_nhpp(*params, total_hours=total_h, n_samples=len(ev_s))
        if len(sim) < 3:
            continue
        ts = np.diff(sim) * 3600.0
        M_c.append(memory(ts)); B_c.append(burstiness(ts)); ML_c.append(memory_log(ts))
    M_c, B_c = np.array(M_c, float), np.array(B_c, float)
    M_c, B_c = M_c[np.isfinite(M_c)], B_c[np.isfinite(B_c)]
    ML_c = np.array(ML_c, float); ML_c = ML_c[np.isfinite(ML_c)]
    return dict(dataset=name, user_id=uid, n_events=len(ev_s), B=B, B_n=Bn, M=M,
                M_shuffle_q95=np.nanpercentile(M_sh, 95), p_M_shuffle=np.nanmean(M_sh >= M),
                M_cnhpp_median=np.median(M_c) if len(M_c) else np.nan,
                p_M_cnhpp=np.mean(M_c >= M) if len(M_c) else np.nan,
                B_cnhpp_median=np.median(B_c) if len(B_c) else np.nan,
                p_B_cnhpp=np.mean(B_c >= B) if len(B_c) else np.nan,
                M_log=ML, p_Mlog_shuffle=np.nanmean(ML_sh >= ML),
                M_log_cnhpp_median=np.median(ML_c) if len(ML_c) else np.nan,
                p_Mlog_cnhpp=np.mean(ML_c >= ML) if len(ML_c) else np.nan, R=len(M_c))


def build_jobs(R, n_shuffle):
    msgs = {c: load_messages(c) for c in ("dana", "vagarenfe")}
    seeds = iter(np.random.SeedSequence(SEED).generate_state(2000))
    jobs = []
    for name, comm, (dmin, dmax), min_msgs, csv in DATASETS:
        df = msgs[comm]
        if dmin:
            df = df[df["date"] >= dmin]
        if dmax:
            df = df[df["date"] <= dmax]                 # same string filter as the notebooks
        counts = df["sender_id"].value_counts()
        users = set(counts[counts > min_msgs].index.astype("int64"))
        res = pd.read_csv(os.path.join(RES, csv))
        assert users == set(res["user_id"]), f"{name}: user set differs from {csv}"
        for _, row in res.iterrows():
            uid = int(row["user_id"])
            dates = df.loc[df["sender_id"] == uid, "date"].sort_values()
            ev_s = (dates - dates.iloc[0]).dt.total_seconds().to_numpy()
            params = (float(row["rho_a"]), float(row["Nw"]), parse_array(row["pd_t"]),
                      parse_array(row["pw_t"]), parse_array(row["p_Na"]))
            jobs.append((name, uid, ev_s, params, R, n_shuffle, int(next(seeds))))
    return jobs


def summarise(out):
    print(f"\n{'dataset':12s} {'n':>3s} {'B med':>6s} {'B_n med':>7s} {'M med':>6s} "
          f"{'M>shuf':>7s} {'M>CNHPP':>8s} {'B>CNHPP':>8s}  (users with p < 0.05)")
    for name, g in out.groupby("dataset", sort=False):
        print(f"{name:12s} {len(g):3d} {g.B.median():6.2f} {g.B_n.median():7.2f} {g.M.median():6.2f} "
              f"{(g.p_M_shuffle < 0.05).sum():4d}/{len(g):<3d}{(g.p_M_cnhpp < 0.05).sum():4d}/{len(g):<3d}"
              f"{(g.p_B_cnhpp < 0.05).sum():4d}/{len(g):<3d}  "
              f"[CNHPP medians: B {g.B_cnhpp_median.median():.2f}, M {g.M_cnhpp_median.median():.2f}]")
        print(f"{'':12s}     M_log med {g.M_log.median():.2f} (CNHPP {g.M_log_cnhpp_median.median():.2f}); "
              f"M_log>shuf {(g.p_Mlog_shuffle < 0.05).sum()}/{len(g)}, M_log>CNHPP {(g.p_Mlog_cnhpp < 0.05).sum()}/{len(g)}")


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
    for ax, (name, g) in zip(axes, out.groupby("dataset", sort=False)):
        ax.scatter(g.M_cnhpp_median, g.B_cnhpp_median, s=18, c="#56B4E9", label="CNHPP (median)", zorder=2)
        ax.scatter(g.M, g.B, s=18, c="#D55E00", label="Empirical", zorder=3)
        for _, r in g.iterrows():
            ax.plot([r.M_cnhpp_median, r.M], [r.B_cnhpp_median, r.B], c="0.8", lw=0.6, zorder=1)
        ax.axvline(0, c="0.5", lw=0.6, ls=":"); ax.axhline(0, c="0.5", lw=0.6, ls=":")
        ax.set_title(name); ax.set_xlabel("Memory coefficient M")
    axes[0].set_ylabel("Burstiness B"); axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight")


def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_shuffle = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else os.cpu_count()
    jobs = build_jobs(R, n_shuffle)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        out = pd.DataFrame(list(ex.map(analyse_user, jobs)))
    out.to_csv(os.path.join(RES, "user_burstiness_memory.csv"), index=False)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    plot(out, os.path.join(HERE, "figures", "user_burstiness_memory.pdf"))
    summarise(out)


if __name__ == "__main__":
    main()
