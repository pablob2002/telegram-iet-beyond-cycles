"""Univariate Hawkes process with a cyclic, piecewise-constant baseline.

    lambda(t) = mu0 * b(t) + n * beta * sum_{t_i < t} exp(-beta (t - t_i))

* time unit: hours; b(t) is constant on 1-hour bins: b(t) = b[floor(t)]
  (b is built by the caller, e.g. weekly profile x community activity, mean ~ 1)
* n = alpha / beta is the branching ratio (expected offspring per event), 0 <= n < 1
* the first event is taken as given (likelihood on (t_1, t_N], conditional on t_1),
  as in the CNHPP fits, which also start at each user's first message.

Functions: fit_null, fit_hawkes, rescaled_iets, simulate.
"""
import numpy as np
from numba import njit
from scipy.optimize import minimize


# ---------------------------------------------------------------- baseline
@njit(cache=True)
def _B(x, b, Cb):
    """Integral of b from 0 to x (b piecewise constant on unit bins, Cb = cumsum)."""
    k = int(np.floor(x))
    if k >= len(b):
        k = len(b) - 1
    return Cb[k] + b[k] * (x - k)


def cumbase(b):
    return np.concatenate([[0.0], np.cumsum(b)])[:-1]


# ---------------------------------------------------------------- likelihood
@njit(cache=True)
def _negloglik(mu0, n, beta, t, b, Cb):
    N = len(t)
    A = 0.0                 # sum_{j<i} exp(-beta (t_i - t_j))
    ll = 0.0
    for i in range(1, N):
        A = np.exp(-beta * (t[i] - t[i - 1])) * (1.0 + A)
        lam = mu0 * b[int(t[i])] + n * beta * A
        if lam <= 0.0:
            return 1e300
        ll += np.log(lam)
    comp = mu0 * (_B(t[N - 1], b, Cb) - _B(t[0], b, Cb))
    for i in range(N - 1):
        comp += n * (1.0 - np.exp(-beta * (t[N - 1] - t[i])))
    return comp - ll


def _unpack(theta):
    return np.exp(theta[0]), 1.0 / (1.0 + np.exp(-theta[1])), np.exp(theta[2])


def fit_null(t, b, Cb):
    """Inhomogeneous Poisson mu0*b(t) (n = 0): closed-form MLE."""
    N = len(t)
    I = _B(t[-1], b, Cb) - _B(t[0], b, Cb)
    mu0 = (N - 1) / I
    ll = np.sum(np.log(mu0 * b[t[1:].astype(np.int64)])) - mu0 * I
    return dict(mu0=mu0, n=0.0, beta=np.nan, loglik=ll)


BETA_STARTS = (60.0, 6.0, 1.0, 0.1)      # 1/beta = 1 min, 10 min, 1 h, 10 h
N_STARTS = (0.2, 0.6)


def fit_hawkes(t, b, Cb, beta_starts=BETA_STARTS, n_starts=N_STARTS):
    null = fit_null(t, b, Cb)
    best = None
    for b0 in beta_starts:
        for n0 in n_starts:
            th0 = np.array([np.log(null["mu0"] * (1 - n0)), np.log(n0 / (1 - n0)), np.log(b0)])
            f = lambda th: _negloglik(*_unpack(th), t, b, Cb)
            r = minimize(f, th0, method="L-BFGS-B",
                         bounds=[(-20, 10), (-12, 12), (np.log(1e-3), np.log(3600.0))])
            if best is None or r.fun < best.fun:
                best = r
    mu0, n, beta = _unpack(best.x)
    return dict(mu0=mu0, n=n, beta=beta, loglik=-best.fun, converged=bool(best.success))


# ---------------------------------------------------------------- goodness of fit
@njit(cache=True)
def rescaled_iets(mu0, n, beta, t, b, Cb):
    """Time-rescaling: Lambda(t_i) - Lambda(t_{i-1}); ~ iid Exp(1) if the model is right."""
    N = len(t)
    z = np.empty(N - 1)
    S = 0.0                 # sum_{j<=i-1} exp(-beta (t_{i-1} - t_j))
    for i in range(1, N):
        S = 1.0 if i == 1 else np.exp(-beta * (t[i - 1] - t[i - 2])) * S + 1.0
        dt = t[i] - t[i - 1]
        z[i - 1] = mu0 * (_B(t[i], b, Cb) - _B(t[i - 1], b, Cb)) + n * S * (1.0 - np.exp(-beta * dt))
    return z


# ---------------------------------------------------------------- simulation
@njit(cache=True)
def simulate(mu0, n, beta, b, t0, t1, seed, max_events=200000):
    """Ogata thinning on (t0, t1], with an event at t0 (as in the fits)."""
    np.random.seed(seed)
    bmax = b.max()
    out = np.empty(max_events)
    out[0] = t0
    k = 1
    s, S = t0, 1.0           # S = sum exp(-beta (s - t_j)) over past events
    while k < max_events:
        lam_bar = mu0 * bmax + n * beta * S
        w = np.random.exponential(1.0 / lam_bar)
        s += w
        if s > t1:
            break
        S *= np.exp(-beta * w)
        lam = mu0 * b[int(s)] + n * beta * S
        if np.random.random() * lam_bar <= lam:
            out[k] = s
            k += 1
            S += 1.0
    return out[:k]
