import numpy as np
import pytest

from simulation.population import User, UserPopulation
from simulation.accountants.standard_dp import StandardDPAccountant
from simulation.environment import SimulationEnvironment
from simulation.records import SimulationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_population(
    n: int,
    epsilon_budget: float,
    delta_budget: float = 1.0,
    conversion_rate: float = 0.3,
) -> UserPopulation:
    users = [
        User(
            user_id=i,
            epsilon_budget=epsilon_budget,
            delta_budget=delta_budget,
            conversion_rate=conversion_rate,
        )
        for i in range(n)
    ]
    return UserPopulation(users)


def make_env(
    population: UserPopulation,
    epsilon_per_round: float = 0.1,
    threshold: float = 3.0,
    num_rounds: int = 10,
    seed: int = 42,
    stop_at_active_fraction: float = 0.0,
) -> SimulationEnvironment:
    return SimulationEnvironment(
        population=population,
        accountant=StandardDPAccountant(epsilon_per_round=epsilon_per_round),
        threshold=threshold,
        query_epsilon=epsilon_per_round,
        num_rounds=num_rounds,
        rng=np.random.default_rng(seed),
        stop_at_active_fraction=stop_at_active_fraction,
    )


# ---------------------------------------------------------------------------
# User dropping
# ---------------------------------------------------------------------------

def test_users_drop_after_budget_exhausted():
    """Users whose epsilon_budget equals one round's cost are dropped after round 1."""
    pop = make_population(n=10, epsilon_budget=0.1, conversion_rate=0.5)
    result = make_env(pop, epsilon_per_round=0.1, num_rounds=5).run()

    assert result.rounds[0].users_dropped == 10
    assert result.rounds[0].num_active_users == 0
    assert result.total_queries_answered == 1


def test_users_survive_all_rounds_with_large_budget():
    """With ample budget, all users should remain active for every round."""
    pop = make_population(n=10, epsilon_budget=100.0, conversion_rate=0.3)
    result = make_env(pop, epsilon_per_round=0.1, num_rounds=10).run()

    assert result.total_queries_answered == 10
    assert result.rounds[-1].num_active_users == 10
    assert result.final_active_fraction == 1.0


def test_staggered_dropout_order():
    """Users with smaller budgets drop before users with larger budgets."""
    users = [
        User(user_id=i, epsilon_budget=0.1 * (i + 1), delta_budget=1.0, conversion_rate=0.1)
        for i in range(5)
    ]
    pop = UserPopulation(users)
    result = make_env(pop, epsilon_per_round=0.1, num_rounds=10).run()

    # After round 1, user 0 (budget=0.1) should be dropped; user 4 (budget=0.5) survives
    active_counts = [r.num_active_users for r in result.rounds]
    assert active_counts[0] < 5  # at least user 0 dropped
    assert active_counts == sorted(active_counts, reverse=True)  # monotonically non-increasing


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

def test_stops_when_all_users_depleted():
    """Simulation terminates before num_rounds if all users drop out."""
    pop = make_population(n=3, epsilon_budget=0.1, conversion_rate=0.9)
    result = make_env(pop, epsilon_per_round=0.1, num_rounds=100).run()

    assert result.total_queries_answered < 100


def test_stop_at_active_fraction():
    """Simulation stops once active fraction falls to or below the threshold."""
    users = [
        User(user_id=i, epsilon_budget=0.1 * (i + 1), delta_budget=1.0, conversion_rate=0.2)
        for i in range(10)
    ]
    pop = UserPopulation(users)
    result = make_env(
        pop, epsilon_per_round=0.1, num_rounds=100, stop_at_active_fraction=0.5
    ).run()

    assert result.total_queries_answered < 100
    # final active fraction should be at or below the stop threshold
    assert result.final_active_fraction <= 0.5


# ---------------------------------------------------------------------------
# Record structure
# ---------------------------------------------------------------------------

def test_round_record_fields():
    """Every RoundRecord should have consistent types and non-negative counts."""
    pop = make_population(n=8, epsilon_budget=10.0, conversion_rate=0.4)
    result = make_env(pop, num_rounds=5).run()

    for i, rec in enumerate(result.rounds):
        assert rec.round_num == i
        assert isinstance(rec.is_top, bool)
        assert rec.private_access_used is True
        assert rec.num_active_users >= 0
        assert rec.true_conversion_count >= 0
        assert rec.users_dropped >= 0
        assert all(eps >= 0.0 for eps in rec.epsilon_remaining_dist)


def test_num_active_users_is_monotone():
    """Active user count must not increase between rounds."""
    pop = make_population(n=20, epsilon_budget=1.0, conversion_rate=0.5)
    result = make_env(pop, epsilon_per_round=0.1, num_rounds=15).run()

    counts = [r.num_active_users for r in result.rounds]
    assert counts == sorted(counts, reverse=True)


def test_epsilon_remaining_decreases():
    """Mean remaining epsilon budget should decrease or stay flat over rounds."""
    pop = make_population(n=20, epsilon_budget=5.0, conversion_rate=0.2)
    result = make_env(pop, epsilon_per_round=0.5, num_rounds=8).run()

    means = result.avg_epsilon_remaining_by_round()
    for a, b in zip(means, means[1:]):
        assert a >= b - 1e-9  # non-increasing (with float tolerance)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_gives_same_result():
    """Two runs with the same seed must produce identical results."""
    def run_once(seed):
        pop = make_population(n=15, epsilon_budget=2.0, conversion_rate=0.35)
        return make_env(pop, num_rounds=10, seed=seed).run()

    r1 = run_once(99)
    r2 = run_once(99)
    assert [rec.is_top for rec in r1.rounds] == [rec.is_top for rec in r2.rounds]
    assert [rec.true_conversion_count for rec in r1.rounds] == \
           [rec.true_conversion_count for rec in r2.rounds]


def test_different_seeds_differ():
    """Different seeds should (almost certainly) produce different outputs."""
    def run_once(seed):
        pop = make_population(n=50, epsilon_budget=10.0, conversion_rate=0.3)
        return make_env(pop, num_rounds=20, seed=seed).run()

    r1 = run_once(1)
    r2 = run_once(2)
    tops1 = [rec.is_top for rec in r1.rounds]
    tops2 = [rec.is_top for rec in r2.rounds]
    assert tops1 != tops2


# ---------------------------------------------------------------------------
# Conversion rate validation
# ---------------------------------------------------------------------------

def test_invalid_conversion_rate_raises():
    """Users with conversion_rate outside [0, 1] should raise on construction."""
    with pytest.raises(ValueError):
        User(user_id=0, epsilon_budget=1.0, delta_budget=1.0, conversion_rate=1.5)


def test_zero_conversion_rate_no_tops_likely():
    """With zero conversion rate, true_count should always be 0."""
    pop = make_population(n=10, epsilon_budget=10.0, conversion_rate=0.0)
    result = make_env(pop, threshold=0.5, num_rounds=20, seed=0).run()
    assert all(r.true_conversion_count == 0 for r in result.rounds)
