import numpy as np

from .population import UserPopulation
from .query import compute_round_conversions, laplace_threshold_query
from .accountants.base import PrivacyAccountant
from .records import RoundRecord, SimulationResult


class SimulationEnvironment:
    """
    Runs the attribution-style threshold monitoring simulation.

    Implements the round-by-round mechanics from the evaluation plan:
      1. Determine attributed conversions for all active users.
      2. Evaluate the Laplace threshold query.
      3. Apply privacy accounting (method-dependent).
      4. Drop users whose budget is now depleted (hidden from server).
      5. Record round metrics.

    The accountant is injected, so the same environment can run Standard IDP,
    Adaptive TCT, or any future method without changing this class.
    """

    def __init__(
        self,
        population: UserPopulation,
        accountant: PrivacyAccountant,
        threshold: float,
        query_epsilon: float,
        num_rounds: int,
        rng: np.random.Generator,
        stop_at_active_fraction: float = 0.0,
    ):
        """
        Args:
            population: user population with initial budgets and conversion rates
            accountant: privacy accountant that determines per-user costs per round
            threshold: T — the value A_r + Lap noise must reach to count as a top
            query_epsilon: epsilon parameter for the Laplace mechanism (controls
                           noise magnitude; also the per-round privacy cost for
                           standard IDP)
            num_rounds: maximum number of rounds R to simulate
            rng: seeded random number generator for reproducibility
            stop_at_active_fraction: stop early once active users fall below this
                                     fraction of the original population (0.0 = run
                                     until all users are gone or R rounds complete)
        """
        self.population = population
        self.accountant = accountant
        self.threshold = threshold
        self.query_epsilon = query_epsilon
        self.num_rounds = num_rounds
        self.rng = rng
        self.stop_at_active_fraction = stop_at_active_fraction

    def run(self) -> SimulationResult:
        total_users = self.population.num_total
        records: list[RoundRecord] = []

        for r in range(self.num_rounds):
            active = self.population.active_users
            if not active:
                break
            if self.population.active_fraction <= self.stop_at_active_fraction:
                break

            # Step 1: sample per-user conversion indicators for this round
            true_count, contributions = compute_round_conversions(active, self.rng)

            # Step 2: evaluate the threshold query with Laplace noise
            is_top, noisy_count = laplace_threshold_query(
                true_count, self.threshold, self.query_epsilon, self.rng
            )

            # Step 3: charge each active user according to the accountant
            for i, user in enumerate(active):
                eps_cost, delta_cost = self.accountant.charge(
                    user=user,
                    is_top=is_top,
                    contributed=contributions[i],
                    round_num=r,
                )
                user.epsilon_spent += eps_cost
                user.delta_spent += delta_cost

            self.accountant.on_round_end(r, is_top, active)

            # Step 4: drop depleted users (server does not observe this)
            dropped = self.population.drop_depleted()

            # Step 5: record post-drop state
            records.append(RoundRecord(
                round_num=r,
                is_top=is_top,
                private_access_used=True,
                num_active_users=self.population.num_active,
                true_conversion_count=true_count,
                noisy_conversion_count=noisy_count,
                users_dropped=dropped,
                epsilon_remaining_dist=self.population.epsilon_remaining_distribution(),
            ))

        return SimulationResult(
            rounds=records,
            total_users=total_users,
            total_queries_answered=len(records),
        )
