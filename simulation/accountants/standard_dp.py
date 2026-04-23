from simulation.population import User
from .base import PrivacyAccountant


class StandardDPAccountant(PrivacyAccountant):
    """
    Standard DP composition accountant.

    Non-independent mode (default): charges every active user (epsilon_per_round,
    delta_per_round) on every round, regardless of individual contribution.
    This is the naive sequential composition baseline.

    Independent mode: charges only when contributed=True — i.e., the user
    personally contributed to this round's query result.  Privacy cost
    concentrates on users who are frequently queried.

    For Laplace mechanism queries, delta_per_round=0.0 (pure DP).

    Args:
        epsilon_per_round:  ε charged per round (when contributing, if independent)
        delta_per_round:    δ charged per round (default 0.0)
        independent:        if True, only charge users who contributed this round
    """

    def __init__(
        self,
        epsilon_per_round: float,
        delta_per_round: float = 0.0,
        independent: bool = False,
    ):
        self.epsilon_per_round = epsilon_per_round
        self.delta_per_round = delta_per_round
        self.independent = independent

    def charge(
        self,
        user: User,
        is_top: bool,
        contributed: bool,
        round_num: int,
    ) -> tuple[float, float]:
        if self.independent and not contributed:
            return (0.0, 0.0)
        return (self.epsilon_per_round, self.delta_per_round)
