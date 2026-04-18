from abc import ABC, abstractmethod

from simulation.population import User


class PrivacyAccountant(ABC):
    """
    Interface for privacy accountants.

    An accountant determines how much (epsilon, delta) budget to charge each
    user per round. Different accountants encode different privacy analyses:
    - StandardIDPAccountant: charges epsilon per round unconditionally.
    - AdaptiveTCTAccountant (Step 2): charges based on target hits and the
      dynamic alpha schedule (Theorem 3 / Algorithms 4-8).

    The charge() method is called once per active user per round, before
    budget is deducted. on_round_end() is called once per round after all
    charges have been applied, for accountants that maintain state across rounds.
    """

    @abstractmethod
    def charge(
        self,
        user: User,
        is_top: bool,
        contributed: bool,
        round_num: int,
    ) -> tuple[float, float]:
        """
        Compute the (epsilon_cost, delta_cost) to deduct from this user's
        budget for the current round.

        Args:
            user: the user being charged (read its current state if needed)
            is_top: whether this round produced a target hit
            contributed: whether this user generated a conversion this round
                         (ground-truth; used by TCT to identify ContributesToTCTTop)
            round_num: zero-based round index
        """
        ...

    def on_round_end(
        self,
        round_num: int,
        is_top: bool,
        active_users: list[User],
    ) -> None:
        """
        Hook called once per round after all per-user charges are applied.
        Override in stateful accountants (e.g. Adaptive TCT) that need to
        update global counters or schedules.
        """
        pass
