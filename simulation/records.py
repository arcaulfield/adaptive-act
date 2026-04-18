from dataclasses import dataclass, field
import numpy as np


@dataclass
class RoundRecord:
    """Per-round snapshot of simulation state, recorded after user dropping."""
    round_num: int
    is_top: bool                          # did the noisy query cross the threshold?
    private_access_used: bool             # always True: Laplace mechanism always queries data
    num_active_users: int                 # active user count after dropping depleted users
    true_conversion_count: int            # A_r ground truth (hidden from server)
    noisy_conversion_count: float         # A_r + Lap noise (what the server observes)
    users_dropped: int                    # users removed this round due to budget depletion
    epsilon_remaining_dist: list[float]   # per-user remaining epsilon among active users


@dataclass
class SimulationResult:
    """Aggregated output of a full simulation run."""
    rounds: list[RoundRecord]
    total_users: int
    total_queries_answered: int

    @property
    def final_active_fraction(self) -> float:
        if not self.rounds:
            return 1.0
        return self.rounds[-1].num_active_users / self.total_users

    def avg_epsilon_remaining_by_round(self) -> list[float]:
        """
        Mean remaining epsilon budget across active users at end of each round.
        Used for the H1 "Queries Answered vs. Privacy Loss" plot.
        """
        return [
            float(np.mean(r.epsilon_remaining_dist)) if r.epsilon_remaining_dist else 0.0
            for r in self.rounds
        ]

    def active_fraction_by_round(self) -> list[float]:
        """
        Fraction of the original population still active at end of each round.
        Used for the H1 "User Retention Curve" plot.
        """
        return [r.num_active_users / self.total_users for r in self.rounds]

    def top_rounds(self) -> list[int]:
        """Round indices where a target hit occurred."""
        return [r.round_num for r in self.rounds if r.is_top]

    def empirical_hit_rate(self) -> float:
        """Fraction of rounds that produced a target hit."""
        if not self.rounds:
            return 0.0
        return sum(r.is_top for r in self.rounds) / len(self.rounds)
