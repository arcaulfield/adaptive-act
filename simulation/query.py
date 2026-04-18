import numpy as np

from .population import User


def compute_round_conversions(
    active_users: list[User],
    rng: np.random.Generator,
) -> tuple[int, list[bool]]:
    """
    Samples each active user's conversion indicator for one round.

    Each user independently generates a conversion with probability equal to
    their conversion_rate (Bernoulli model). Because each user contributes at
    most 1, the L1 sensitivity of the total count A_r is 1, which is required
    for Lap(1/epsilon) to achieve epsilon-DP.

    Returns:
        total_count: A_r = sum of per-user indicators
        contributions: per-user boolean list (ground truth; hidden from server)
    """
    contributions = [bool(rng.random() < u.conversion_rate) for u in active_users]
    return int(sum(contributions)), contributions


def laplace_threshold_query(
    true_count: int,
    threshold: float,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[bool, float]:
    """
    Evaluates the threshold query q_r(D_r) = 1{A_r + Lap(1/epsilon) >= threshold}.

    This is Algorithm 3 from the paper (TCT Mechanism). The Laplace noise
    calibrated to sensitivity 1 gives epsilon-DP.

    Returns:
        is_top: whether the noisy count crossed the threshold (target hit)
        noisy_count: the noisy value observed by the server
    """
    noise = rng.laplace(loc=0.0, scale=1.0 / epsilon)
    noisy_count = true_count + noise
    is_top = bool(noisy_count >= threshold)
    return is_top, noisy_count
