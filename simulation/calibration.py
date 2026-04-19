"""
Threshold calibration for the TCT simulation.

q in the TCT paper (Definition 1) is:

    q = inf_D  Pr_{r ~ B(D)}[r ∈ ⊤]

For the Laplace threshold mechanism (Algorithm 3), with sensitivity-1 queries,
this becomes the probability that the *noisy* count crosses threshold T:

    q = P(true_count + Lap(1/ε) ≥ T)

minimized over neighboring datasets (worst-case: removing the single highest
contributor reduces the count by at most 1).

Correct calibration must therefore use the NOISY count distribution, not the
noiseless one. Setting T at the (1−q) quantile of noiseless counts produces a
q_actual that is 1.6–4.8× larger than intended (the Laplace noise adds extra
mass above the threshold), inflating K_k and degrading the tightness of H2.

Usage:
    from simulation.calibration import calibrate_threshold

    T, q_empirical = calibrate_threshold(
        population=pop,
        epsilon=0.1,
        target_q=0.3,
        n_pilot=2000,
        rng=np.random.default_rng(0),
    )
"""

from __future__ import annotations

import numpy as np

from .population import UserPopulation
from .query import compute_round_conversions


def calibrate_threshold(
    population: UserPopulation,
    epsilon: float,
    target_q: float,
    n_pilot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Find threshold T such that P(true_count + Lap(1/ε) ≥ T) ≈ target_q.

    Sets T at the (1−target_q) empirical quantile of the noisy count
    distribution, so that exactly target_q fraction of pilot rounds produce a
    noisy crossing. Passing this T and target_q to the TCT accountant ensures
    the accountant's q parameter matches the mechanism's actual hit probability.

    Args:
        population:  the user population whose conversion rates govern true counts
        epsilon:     the Laplace mechanism's ε (noise scale = 1/ε)
        target_q:    desired empirical hit rate in (0, 1)
        n_pilot:     number of pilot rounds used to estimate the quantile
        rng:         seeded generator; creates a fresh one if None

    Returns:
        threshold:    T — the noisy-count threshold to pass to SimulationEnvironment
        q_empirical:  fraction of pilot rounds where noisy_count ≥ T (≈ target_q)
    """
    if rng is None:
        rng = np.random.default_rng()
    if not 0 < target_q < 1:
        raise ValueError(f"target_q must be in (0, 1), got {target_q}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    noisy_counts = np.empty(n_pilot)
    for i in range(n_pilot):
        true_count, _ = compute_round_conversions(population.active_users, rng)
        noise = rng.laplace(0.0, 1.0 / epsilon)
        noisy_counts[i] = true_count + noise

    threshold = float(np.quantile(noisy_counts, 1.0 - target_q))
    q_empirical = float(np.mean(noisy_counts >= threshold))
    return threshold, q_empirical
