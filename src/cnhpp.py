"""CNHPP estimation code (Malmgren et al. 2008), extracted VERBATIM from
notebooks/03_cnhpp/01_CNHPP_DANA.ipynb so that new analyses use exactly the same code
as the cached results in results/cnhpp/. Do not edit the functions here without
re-running the CNHPP notebooks.

`trapz` is not imported in any cell of the notebook (it came from a deleted cell);
scipy.integrate.trapezoid implements the same trapezoidal rule.
"""
import random

import numpy as np
import pandas as pd

try:
    from scipy.integrate import trapezoid as trapz
except ImportError:  # older SciPy
    from scipy.integrate import trapz

def compute_model_params(events, active_config):
    """
    events: array de tiempos en HORAS.
    active_config: bool array N. True = mismo AI que el evento anterior.
    """
    iets = np.diff(events)
    iet_active = iets[active_config[1:]]
    rho_a = 1.0 / np.mean(iet_active) if len(iet_active) > 0 else 0

    n_active_intervals = np.sum(~active_config[1:]) + 1
    total_weeks = (events[-1] - events[0]) / (7 * 24)
    Nw = n_active_intervals / total_weeks if total_weeks > 0 else 0

    starts_idx = np.where(~active_config[1:])[0] + 1
    start_times = np.concatenate([events[:1], events[starts_idx]])
    hours    = start_times % 24
    weekdays = (start_times / 24) % 7

    pd_counts, _ = np.histogram(hours,    bins=24, range=(0, 24))
    pw_counts, _ = np.histogram(weekdays, bins=7,  range=(0, 7))
    pd_t = pd_counts / pd_counts.sum() if pd_counts.sum() > 0 else np.zeros(24)
    pw_t = pw_counts / pw_counts.sum() if pw_counts.sum() > 0 else np.zeros(7)
    return rho_a, Nw, pd_t, pw_t


def get_pNa(config):
    """
    Distribución empírica de Na. config: bool array N-1, True = par (i,i+1) en el mismo AI.
    """
    na_counts, count = [], 0
    for c in config:
        if c:
            count += 1
        else:
            na_counts.append(count)
            count = 0
    na_counts.append(count)
    hist = np.bincount(na_counts)
    return hist / hist.sum()


def _init_threshold(iets):
    """
    Calcula el umbral iterativamente: threshold = 0.1/rho_primary = 0.1 * mean_iet * (1+mean_Na).
    """
    pos = iets[iets > 0]
    if len(pos) < 20:
        return np.median(pos)

    mean_iet  = np.mean(pos)
    threshold = 0.1 * mean_iet

    for _ in range(15):
        n_within  = int(np.sum(pos < threshold))
        n_between = int(np.sum(pos >= threshold))
        mean_Na_est   = n_within / max(n_between, 1)
        threshold_new = 0.1 * mean_iet * (1 + mean_Na_est)
        if abs(threshold_new - threshold) < 1e-6 * mean_iet:
            break
        threshold = threshold_new

    return threshold_new


def area_test_statistic(empirical_iet, model_iet, n_bins=100, eps=1e-6):
    emp = np.asarray(empirical_iet)
    mod = np.asarray(model_iet)
    emp = emp[np.isfinite(emp) & (emp > 0)]
    mod = mod[np.isfinite(mod) & (mod > 0)]
    if len(emp) == 0 or len(mod) == 0:
        return np.inf
    lo = max(emp.min(), mod.min(), eps)
    hi = max(emp.max(), mod.max(), lo * 10)
    log_bins = np.logspace(np.log10(lo), np.log10(hi), n_bins)
    ccdf_emp = np.array([np.mean(emp > b) for b in log_bins])
    ccdf_mod = np.array([np.mean(mod > b) for b in log_bins])
    return trapz(np.abs(ccdf_emp - ccdf_mod), np.log(log_bins))


def simulate_cascading_nhpp(rho_a, Nw, pd_t, pw_t, p_Na, total_hours, n_samples=None):
    """
    Simula el cascading NHPP del paper Malmgren et al. (2008).
    Proceso primario: NHPP con tasa rho(t) = Nw * pd(t) * pw(t).
    Proceso secundario: HPP con tasa rho_a dentro de cada active interval.
    """
    pd_t_arr = np.asarray(pd_t, dtype=float)
    pw_t_arr = np.asarray(pw_t, dtype=float)
    pd_t_arr = pd_t_arr / pd_t_arr.sum()
    pw_t_arr = pw_t_arr / pw_t_arr.sum()
    p_Na_norm = np.asarray(p_Na, dtype=float)
    p_Na_norm = p_Na_norm / p_Na_norm.sum()

    mean_Na = np.dot(np.arange(len(p_Na_norm)), p_Na_norm)

    # Si n_samples requiere más eventos de los naturales, extender la ventana temporal
    sim_hours = total_hours
    if n_samples is not None and Nw > 0:
        expected_events = Nw * (total_hours / (7 * 24)) * (1 + mean_Na)
        if expected_events < n_samples:
            sim_hours = total_hours * (n_samples / max(1, expected_events)) * 1.5

    n_weeks = sim_hours / (7 * 24)

    # Distribución conjunta (día, hora): outer product pw_t x pd_t → 168 slots semanales
    joint = np.outer(pw_t_arr, pd_t_arr).flatten()  # shape (7*24,)
    joint /= joint.sum()

    n_bursts = np.random.poisson(max(1.0, Nw * n_weeks))

    # Muestrear semana (uniforme en [0, n_weeks)) y slot (día, hora) de la distribución conjunta
    week_nums = np.random.uniform(0, n_weeks, n_bursts)
    slots = np.random.choice(168, size=n_bursts, p=joint)

    # Convertir a horas absolutas: inicio_semana + día*24 + hora + ruido uniforme
    burst_times = (
        np.floor(week_nums) * 7 * 24        # inicio de la semana entera
        + (slots // 24) * 24                # offset del día (0–6 × 24 h)
        + (slots % 24)                      # offset de la hora (0–23 h)
        + np.random.uniform(0, 1, n_bursts) # ruido uniforme dentro de la hora
    )
    burst_times = np.sort(burst_times)
    burst_times = burst_times[burst_times < sim_hours]

    if rho_a <= 0:
        rho_a = 1.0

    # Generar cascadas: inicio de ráfaga + Na eventos adicionales a tasa rho_a
    events = []
    for bt in burst_times:
        events.append(bt)
        Na = np.random.choice(len(p_Na_norm), p=p_Na_norm)
        t = bt
        for _ in range(Na):
            t += np.random.exponential(1.0 / rho_a)
            events.append(t)

    # IMPORTANTE: ordenar — dos cascadas consecutivas pueden solaparse en tiempo,
    # produciendo IETs negativos si no se ordena antes de np.diff()
    result = np.sort(np.array(events))

    if n_samples is not None:
        return result[:n_samples]
    return result


def simulated_annealing(events_hours, n_iter=5000, T_init=1.0, cooling=0.95, n_sim=None):
    """
    Simulated Annealing para identificar active intervals (Malmgren et al. 2008, SI).

    Correcciones respecto al código original:
      (1) pd_t y pw_t se fijan antes del SA y no se recalculan (SI Simplification #2).
      (2) Solo se proponen flips para pares con τ < threshold (SI Simplification #3).
      (3) cool_every = n_iter // 50 garantiza ~50 pasos de enfriamiento.
          El paper dice "cada 2N iters", lo que requiere n_iter >> 100*N para converger.
          Para n_iter=5000 con N~400, cool_every=100 es equivalente y converge bien.
    """
    N    = len(events_hours)
    iets = np.diff(events_hours)
    if n_sim is None:
        n_sim = max(5000, 10 * N)

    # Corrección #1: fijar pd_t y pw_t de la distribución empírica ANTES del SA
    all_hours    = events_hours % 24
    all_weekdays = (events_hours / 24) % 7
    pd_counts, _ = np.histogram(all_hours,    bins=24, range=(0, 24))
    pw_counts, _ = np.histogram(all_weekdays, bins=7,  range=(0, 7))
    pd_t_fixed = pd_counts / pd_counts.sum() if pd_counts.sum() > 0 else np.ones(24) / 24
    pw_t_fixed = pw_counts / pw_counts.sum() if pw_counts.sum() > 0 else np.ones(7) / 7

    def compute_area(cfg):
        rho_a, Nw, _, _ = compute_model_params(
            events_hours, np.concatenate([[False], cfg])
        )
        p_Na = get_pNa(cfg)
        model_events = simulate_cascading_nhpp(
            rho_a, Nw, pd_t_fixed, pw_t_fixed, p_Na,
            total_hours=events_hours[-1] - events_hours[0],
            n_samples=n_sim
        )
        if len(model_events) > 1:
            return area_test_statistic(iets, np.diff(model_events))
        return np.inf

    threshold = _init_threshold(iets)
    config    = iets < threshold

    # Corrección #2: solo proponer flips para pares con τ < threshold
    mergeable = np.where(iets < threshold)[0]
    if len(mergeable) == 0:
        mergeable = np.arange(N - 1)

    print(f"  Init: {config.mean():.1%} pares mergeados  "
          f"(threshold={threshold:.4f}h = {threshold*60:.1f}min, "
          f"candidatos={len(mergeable)}/{N-1})")

    T           = T_init
    best_config = config.copy()
    A_current   = compute_area(config)
    A_best      = A_current

    # Corrección #3: ~50 pasos de enfriamiento garantizados en n_iter iteraciones
    cool_every = max(1, n_iter // 50)

    for iteration in range(n_iter):
        idx        = mergeable[random.randint(0, len(mergeable) - 1)]
        new_config = config.copy()
        new_config[idx] = not new_config[idx]

        A_new = compute_area(new_config)
        delta = A_new - A_current
        if delta < 0 or random.random() < np.exp(-delta / T):
            config    = new_config
            A_current = A_new
            if A_new < A_best:
                A_best      = A_new
                best_config = new_config.copy()

        if (iteration + 1) % cool_every == 0:
            T *= cooling

    return best_config, A_best


def week_reshuffle_iets(dates, n_shuffles=200, seed=42):
    """
    Reshuffle messages between weeks keeping (weekday, time_of_day) fixed.

    Uses permutation (without replacement) of week labels so the marginal
    day-of-week and time-of-day distributions are exactly preserved.
    Returns (original_iets, list_of_shuffled_iets).
    """
    rng = np.random.default_rng(seed)
    dates = pd.Series(dates).dropna().sort_values().reset_index(drop=True)
    if len(dates) < 10:
        return None, None

    # Week index relative to the first message (Monday-aligned)
    first_monday = dates.iloc[0] - pd.Timedelta(days=dates.iloc[0].weekday())
    week_num = ((dates - first_monday).dt.total_seconds() // (7 * 86400)).astype(int).values
    weekday  = dates.dt.weekday.values                                     # 0–6
    tod_sec  = (dates.dt.hour * 3600 + dates.dt.minute * 60 + dates.dt.second).values

    unique_weeks = np.unique(week_num)
    n_weeks = len(unique_weeks)

    # Original IETs in seconds
    original_ts   = dates.values.astype('int64') / 1e9
    original_iets = np.diff(np.sort(original_ts))
    original_iets = original_iets[original_iets > 0]

    shuffled_iets_list = []
    for _ in range(n_shuffles):
        # Permute week labels (without replacement to keep week-count distribution)
        perm_weeks = rng.permutation(week_num)
        new_ts = (perm_weeks * 7 * 86400 + weekday * 86400 + tod_sec).astype(float)
        new_ts_sorted = np.sort(new_ts)
        new_iets = np.diff(new_ts_sorted)
        new_iets = new_iets[new_iets > 0]
        shuffled_iets_list.append(new_iets)

    return original_iets, shuffled_iets_list
