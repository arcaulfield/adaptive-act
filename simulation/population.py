from dataclasses import dataclass, field


@dataclass
class User:
    """
    A single participant in the simulation.

    conversion_rate must be in [0, 1]: probability of generating an attributed
    conversion in a given round. Capped at 1.0 so the query sensitivity stays
    equal to 1 and Lap(1/epsilon) is the correct noise magnitude.
    """
    user_id: int
    epsilon_budget: float
    delta_budget: float
    conversion_rate: float
    active: bool = field(default=True, init=True)
    epsilon_spent: float = field(default=0.0, init=False)
    delta_spent: float = field(default=0.0, init=False)

    def __post_init__(self):
        if not 0.0 <= self.conversion_rate <= 1.0:
            raise ValueError(
                f"User {self.user_id}: conversion_rate must be in [0, 1], "
                f"got {self.conversion_rate}. Cap LogNormal samples at 1.0 before "
                "constructing users."
            )

    @property
    def epsilon_remaining(self) -> float:
        return self.epsilon_budget - self.epsilon_spent

    @property
    def delta_remaining(self) -> float:
        return self.delta_budget - self.delta_spent

    @property
    def is_depleted(self) -> bool:
        return self.epsilon_remaining <= 0.0 or self.delta_remaining < 0.0


class UserPopulation:
    """Manages the full set of users across rounds."""

    def __init__(self, users: list[User]):
        self._users = users

    @property
    def all_users(self) -> list[User]:
        return self._users

    @property
    def active_users(self) -> list[User]:
        return [u for u in self._users if u.active]

    @property
    def num_active(self) -> int:
        return sum(1 for u in self._users if u.active)

    @property
    def num_total(self) -> int:
        return len(self._users)

    @property
    def active_fraction(self) -> float:
        if not self._users:
            return 0.0
        return self.num_active / len(self._users)

    def drop_depleted(self) -> int:
        """
        Deactivates any user whose budget is now at or below zero.
        This information is hidden from the server; only the simulation
        infrastructure calls this.
        Returns the number of users dropped this call.
        """
        dropped = 0
        for u in self._users:
            if u.active and u.is_depleted:
                u.active = False
                dropped += 1
        return dropped

    def epsilon_remaining_distribution(self) -> list[float]:
        """Returns remaining epsilon budgets for all currently active users."""
        return [u.epsilon_remaining for u in self._users if u.active]
