# Adaptive TCT Simulation

Simulation harness for **"Adaptive Target-Charging with Privacy Filters and Individual Accounting"** (NeurIPS 2026 submission).

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `jupyterlab`, `pytest`.

## Reproducing Paper Figures

Pre-generated figures are committed to `figures/`. To regenerate them, run the corresponding notebook from the `notebooks/` directory:

```bash
cd notebooks
jupyter lab                            # open interactively, or:
jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb \
    --ExecutePreprocessor.timeout=1200
```

### Main Text

| Paper Figure | Notebook | Output files |
|---|---|---|
| Figure 1 — Q1: User retention & budget | `01_q1_user_retention.ipynb` | `figures/h1_rdp_s{1,2,3}_{retention,budget}.png` |
| Figure 2 — Q2: Adaptivity cost (bar) | `02_q2_adaptivity_cost.ipynb` | `figures/adaptivity_waste_fixed_tau.png` |
| Figure 3 — Q2: Adaptivity cost (curves) | `02_q2_adaptivity_cost.ipynb` | `figures/adaptivity_waste_diff_tau.png` |
| Figure 4 — Q3: Hit rate sensitivity | `03_q3_hit_rate_sensitivity.ipynb` | `figures/h4_hit_rate_s3.png` |

### Appendix

| Figure | Notebook | Output files |
|---|---|---|
| H2: Theoretical vs empirical bounds | `04_q4_bounds.ipynb` | `figures/h2_rdp_bounds.png` |
| H2: Empirical failure rate vs δ | `04_q4_bounds.ipynb` | `figures/h2_rdp_failure_rate.png` |
| Ablation: α sweep (retention + budget) | `05_appendix_ablations.ipynb` | `figures/ablation_alpha_sweep{,_budget}.png` |
| Ablation: Population distribution | `05_appendix_ablations.ipynb` | `figures/ablation_pop_distribution{,_budget}.png` |
| Ablation: δ sweep (retention + budget) | `05_appendix_ablations.ipynb` | `figures/ablation_delta_sweep{,_budget}.png` |

**Note:** `15_h2_rdp.ipynb` runs 1,000,000 Monte Carlo trials and takes ~10 minutes. All other notebooks complete in under 5 minutes each.

## Repository Structure

```
simulation/                  Core simulation package (importable from notebooks/)
  accountants/
    adaptive_tct.py          A-TCT accountant: Zeta-2 and Stitched RDP schedules
                             Implements Algorithm 3 / Corollaries 2 & 4 from the paper
    standard_dp.py           Standard RDP baseline (charges ρ every round)
    standard_tct.py          Non-adaptive TCT baseline
  notebook_utils.py          Shared plot helpers, colors, and style used by all notebooks
  calibration.py             q_from_epsilon(): NotPrior target parameter (Lemma 2.3)
  datasets.py                Synthetic user population (Beta-distributed conversion rates)
  environment.py             Round-by-round simulation mechanics
  query.py                   Laplace threshold query and round conversion counting
  population.py              User state: budget tracking, active-set management

notebooks/                   Paper figure notebooks — run these to reproduce all figures
  01_q1_user_retention.ipynb     Figure 1: RDP composition, three participation scenarios
  02_q2_adaptivity_cost.ipynb    Figures 2–3: cost of using fixed τ vs adaptive stopping
  03_q3_hit_rate_sensitivity.ipynb  Figure 4: half-life advantage scales as q/h
  04_q4_bounds.ipynb             Appendix H2: theoretical vs empirical bound tightness
  05_appendix_ablations.ipynb    Appendix: α, population-distribution, and δ sweeps

figures/                     Pre-generated figures (committed for reviewer convenience)
tests/                       Unit tests for the simulation package
ProjectDocumentation/        Paper drafts, evaluation plan, and figure manifest
```

## Running Tests

```bash
pytest tests/
```

## Key Experimental Parameters

All paper figures use:

| Parameter | Value | Meaning |
|---|---|---|
| ε | 0.01 | Per-query privacy cost |
| α | 75 | RDP order; ρ = αε²/2 = 3.75×10⁻³ |
| δ∞ | 1e-2 | IDP filter failure probability |
| N | 10,000 | Number of simulated users |
| N_rounds | 1,500 | Maximum query rounds |
| Target hit rate | 2% | Fraction of rounds that cross threshold T |
| ε_RDP budget | 1.0 | Per-user privacy budget (in RDP units) |

**Scenarios** (S1–S3) vary participation rate `p` and interleave ratio (fraction of non-TCT rounds):

| Scenario | Participation | Non-TCT rounds | Notes |
|---|---|---|---|
| S1 | 100% | 0% | Pure TCT; maximum advantage |
| S2 | 100% | 50% | Mixed workload |
| S3 | 50% | 50% | Most realistic; paper's primary scenario |
