from simulation.population import User
from .base import PrivacyAccountant


class StandardIDPAccountant(PrivacyAccountant):
    """
    Baseline accountant: standard individual DP composition.

    Charges every active user (epsilon_per_round, delta_per_round) on every
    round they participate in, regardless of whether the round produced a
    target hit. This is the naive sequential composition baseline — it treats
    all queries as equally costly and ignores the threshold structure.

    For Laplace mechanism queries, delta_per_round=0.0 (pure DP), so only the
    epsilon budget is consumed.
    """

    def __init__(self, epsilon_per_round: float, delta_per_round: float = 0.0):
        self.epsilon_per_round = epsilon_per_round
        self.delta_per_round = delta_per_round

    def charge(
        self,
        user: User,
        is_top: bool,
        contributed: bool,
        round_num: int,
    ) -> tuple[float, float]:
        return (self.epsilon_per_round, self.delta_per_round)
