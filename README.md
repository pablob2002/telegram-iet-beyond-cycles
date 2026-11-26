# Temporal dynamics in crisis-driven Telegram communities beyond circadian and weekly rhythms

Code and pseudonymized data for the preprint

> P. Barquín Soria, J. Hernández Tey and E. Cozzo, *Temporal dynamics in crisis-driven
> user-organized Telegram communities beyond circadian and weekly rhythms* (2026). Preprint: TBD.

Universitat de Barcelona Institute of Complex Systems (UBICS) and Heurística Lab, within the
DRI project (Desinformació, Resiliència i Infraestructures).

The repository contains only the analyses used in the paper: inter-event-time (IET)
distributions and truncated power-law fits, the Kaplan–Meier finite-observation-window
correction, the time-rescaling check, and the test of the cascading non-homogeneous Poisson
process (CNHPP) of Malmgren et al. together with the week-reshuffling stationarity test.
It is derived from the master's-thesis repository (TFM, Universitat de Barcelona, 2026), which
also covers information-disorder analyses that are not part of this paper.

## Data

Two public Telegram communities: **@voluntariosdanavalencia** (volunteer coordination after
the October 2024 DANA flood in Valencia) and **@vagarenfe** (Rodalies Renfe commuters,
analyzed in 2024 and from 15 January 2026). The pseudonymized message tables (timestamps, pseudonymous sender ids, sub-channel names)
are not yet included: their release is pending approval by the DRI project. Cached results
in `results/` are included. See [`data/README.md`](data/README.md).

## Layout

```
src/tgdata.py                      load_messages("dana" | "vagarenfe")
src/cnhpp.py                       CNHPP functions extracted verbatim from 03_cnhpp/01_CNHPP_DANA.ipynb
src/hawkes_cyclic.py               Hawkes process with cyclic baseline
scripts/pseudonymize.py            builds data/processed and results/ from the raw exports
data/processed/                    pseudonymized message tables (not yet released)
results/iet/                       Table I (global fits, CIs, goodness of fit)
results/cnhpp/                     cached CNHPP fits and Monte Carlo p-values (pseudonymized)
notebooks/01_iet_distributions/    global and per-channel IET fits, Kaplan–Meier correction
notebooks/02_time_rescaling/       time-rescaling check of the fitted distributions
notebooks/03_cnhpp/                CNHPP fits, Monte Carlo GoF, week reshuffling, burstiness/memory
notebooks/04_hawkes/               Hawkes process with cyclic baseline (src/hawkes_cyclic.py)
```

Each notebook writes its figures to a `figures/` folder next to it. Notebooks are numbered in
run order and must be run from their own folder.

## Reproducing the results

```bash
pip install -r requirements.txt
jupyter lab   # open the notebooks from their folder
```

The CNHPP notebooks (`03_cnhpp/01–03`) load the cached results in `results/cnhpp/` and skip
the expensive estimation. Move those CSVs away to recompute from scratch (several hours per
dataset). Settings used for the cached results:

| Step | Setting |
|---|---|
| Simulated annealing (fit to data) | 5,000 iterations |
| Monte Carlo goodness of fit | M = 100 replicas, full SA refit (5,000 iterations) per replica |
| Week reshuffling (KS diagnostic) | N = 200 permutations per user |
| CNHPP refit on reshuffled data | one permutation per user, 2,000 SA iterations; Monte Carlo M = 100 |
| Active users | DANA > 40 messages; RENFE > 100 messages in the window; reshuffling restricted to users active ≥ 8 days |
| Power-law fits | `powerlaw`, `discrete=True` (1 s timestamp resolution) |
| Table I uncertainties | B = 500 nonparametric bootstraps; GoF: B = 500 semiparametric bootstraps (Clauset et al. 2009), xmin re-estimated in each replica |

## Where each result comes from

| Result | Produced by |
|---|---|
| Table I: global TPL fits (α, λ, 95% bootstrap CIs, KS D, GoF p) | `01_iet_distributions/table1_global_fits.py` → `results/iet/table1_global_fits.csv` |
| Global CCDF comparison figure | `01_iet_distributions/waiting_time_fit_combined.py` |
| Model comparison, goodness of fit, global IETs (full export window) | `01_iet_distributions/01–02_global_IET_*.ipynb` |
| Per-channel exponents (Kaplan–Meier corrected) and figure | `01_iet_distributions/03–04_channel_*.ipynb`, `combine_alpha_channels_numbered.py` |
| Daily activity of @vagarenfe; temporal sections | `01_iet_distributions/04_channel_IET_vagarenfe.ipynb`, `05_vagarenfe_sections.ipynb` |
| Time-rescaling check | `02_time_rescaling/*` |
| Per-user CNHPP fits and Monte Carlo p-values | `03_cnhpp/01–03_CNHPP_*.ipynb` |
| CNHPP failure panel (CCDFs, reshuffled surrogate, A_best) | `03_cnhpp/04_CNHPP_failure_panel.ipynb` |
| Week-reshuffling stationarity test and rejection counts | `03_cnhpp/05_reshuffle_stationarity_analysis.ipynb` |
| Per-user burstiness B and memory M vs. the fitted CNHPP | `03_cnhpp/06_user_burstiness_memory.py` → `results/cnhpp/user_burstiness_memory.csv` |
| Fair original-vs-reshuffled comparison (A from fresh simulations), rescued users | `03_cnhpp/07_reshuffle_fair_comparison.py` → `results/cnhpp/reshuffle_fair_comparison.csv` |
| Convergence of the 2,000-iteration SA fits on reshuffled data (DANA) | `03_cnhpp/08_sa_convergence_check.py` → `results/cnhpp/sa_convergence_check_dana.csv` |
| CNHPP rejection vs number of messages and temporal concentration | `03_cnhpp/09_rejection_power_vs_concentration.py` → `results/cnhpp/rejection_covariates.csv` |
| Hawkes pilot (6 RENFE 2026 users) | `04_hawkes/01_hawkes_pilot_renfe2026.py` → `results/hawkes/pilot_renfe2026.csv` |

## License

Code: MIT (see `LICENSE`). Data: terms to be confirmed (see `data/README.md`).

## Citation

TBD once the preprint is online.
