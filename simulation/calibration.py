"""
Threshold calibration and q computation for the TCT simulation.

  q = 1 / (e^ε + 1)          (Lemma 2.3, Cohen & Lyu 2023)

    q is a property of the ε-DP mechanism alone. For the Laplace threshold
    mechanism, the NotPrior target ⊤ = {TRUE} is always a 1/(e^ε+1)-target,
    regardless of the threshold T or the data distribution. Use q_from_epsilon
    to compute it.

Usage
-----
    from simulation.calibration import q_from_epsilon, set_threshold_for_hit_rate

    q = q_from_epsilon(epsilon)          # e.g. 0.475 for epsilon=0.1
    T, hit_rate = set_threshold_for_hit_rate(
        population=pop,
        epsilon=epsilon,
        target_hit_rate=0.05,            # want 5% of rounds to cross T
        n_pilot=2000,
        rng=np.random.default_rng(0),
    )
    # verify: hit_rate ≈ 0.05, and hit_rate << q

    accountant = AdaptiveTCTAccountant(epsilon=epsilon, delta=delta)
    # accountant.q == q  (set automatically from epsilon)
"""

from __future__ import annotations

import math

import numpy as np

from .population import UserPopulation
from .query import compute_round_conversions


def q_from_epsilon(epsilon: float) -> float:
    """
    Compute q from epsilon via Lemma 2.3 (NotPrior target).

    For any ε-DP mechanism A: X → Y ∪ {⊥}, the set of non-⊥ outcomes Y
    constitutes a 1/(e^ε + 1)-target. For the Laplace threshold mechanism
    with ⊥ = FALSE, this gives the hit probability in the private branch:

        q = 1 / (e^ε + 1)

    For small ε, q → 1/2 (overhead factor ≈ 2).
    For large ε, q decreases (the mechanism is less informative per query).
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    return 1.0 / (math.exp(epsilon) + 1.0)


def set_threshold_for_hit_rate(
    population: UserPopulation,
    epsilon: float,
    target_hit_rate: float,
    n_pilot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """
    Find threshold T such that the empirical noisy hit rate ≈ target_hit_rate.

    Sets T at the (1 − target_hit_rate) quantile of the noisy count
    distribution (true_count + Lap(1/ε)), so that approximately target_hit_rate
    fraction of rounds produce a crossing.

    This is an application-level choice — T controls HOW OFTEN the threshold
    is crossed. It is independent of q, which is fixed by epsilon alone.

    For TCT to be beneficial, set target_hit_rate << q_from_epsilon(epsilon).
    For example, with epsilon=0.1 (q≈0.475), a target_hit_rate of 0.05–0.10
    means ~90–95% of rounds output ⊥ and pay nothing under TCT.

    Args:
        population:       user population whose conversion rates govern counts
        epsilon:          Laplace mechanism ε (noise scale = 1/ε)
        target_hit_rate:  desired fraction of rounds where noisy_count ≥ T
        n_pilot:          number of pilot rounds used to estimate the quantile
        rng:              seeded generator; creates a fresh one if None

    Returns:
        threshold:          T to pass to SimulationEnvironment
        empirical_hit_rate: fraction of pilot rounds where noisy_count ≥ T
    """
    if rng is None:
        rng = np.random.default_rng()
    if not 0 < target_hit_rate < 1:
        raise ValueError(f"target_hit_rate must be in (0, 1), got {target_hit_rate}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    noisy_counts = np.empty(n_pilot)
    for i in range(n_pilot):
        true_count, _ = compute_round_conversions(population.active_users, rng)
        noise = rng.laplace(0.0, 1.0 / epsilon)
        noisy_counts[i] = true_count + noise

    threshold = float(np.quantile(noisy_counts, 1.0 - target_hit_rate))
    empirical_hit_rate = float(np.mean(noisy_counts >= threshold))
    return threshold, empirical_hit_rate
