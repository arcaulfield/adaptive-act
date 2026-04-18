import numpy as np
import pytest

from simulation.datasets import (
    PopulationConfig,
    describe_population,
    generate_synthetic_population,
)
from simulation.population import UserPopulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPARSE = PopulationConfig(
    n_users=2000,
    alpha=0.3,
    beta=3.0,
    epsilon_budget=1.0,
    delta_budget=1e-5,
)

MODERATE = PopulationConfig(
    n_users=2000,
    alpha=0.5,
    beta=2.0,
    epsilon_budget=1.0,
    delta_budget=1e-5,
)


def sample(config: PopulationConfig, seed: int = 0) -> UserPopulation:
    return generate_synthetic_population(config, np.random.default_rng(seed))


# ---------------------------------------------------------------------------
# Population size and structure
# ---------------------------------------------------------------------------

def test_correct_number_of_users():
    pop = sample(SPARSE)
    assert len(pop.all_users) == SPARSE.n_users
    assert pop.num_total == SPARSE.n_users
    assert pop.num_active == SPARSE.n_users


def test_all_users_active_at_creation():
    pop = sample(SPARSE)
    assert all(u.active for u in pop.all_users)


def test_user_ids_are_sequential():
    pop = sample(SPARSE)
    assert [u.user_id for u in pop.all_users] == list(range(SPARSE.n_users))


# ---------------------------------------------------------------------------
# Rate distribution: correctness
# ---------------------------------------------------------------------------

def test_all_rates_in_unit_interval():
    """Beta samples must be in [0, 1] — no capping should be required."""
    pop = sample(SPARSE)
    rates = [u.conversion_rate for u in pop.all_users]
    assert all(0.0 <= r <= 1.0 for r in rates), "Rate outside [0, 1] found"


def test_mean_close_to_theoretical():
    """Sample mean should be within 3 standard errors of E[rate] = alpha/(alpha+beta)."""
    pop = sample(SPARSE, seed=7)
    observed_mean = np.mean([u.conversion_rate for u in pop.all_users])
    expected = SPARSE.expected_rate
    # 3-sigma tolerance for N=2000
    tol = 3 * SPARSE.rate_std / np.sqrt(SPARSE.n_users)
    assert abs(observed_mean - expected) < tol, (
        f"Mean {observed_mean:.4f} too far from theoretical {expected:.4f}"
    )


def test_mean_close_to_theoretical_moderate():
    pop = sample(MODERATE, seed=3)
    observed_mean = np.mean([u.conversion_rate for u in pop.all_users])
    expected = MODERATE.expected_rate
    tol = 3 * MODERATE.rate_std / np.sqrt(MODERATE.n_users)
    assert abs(observed_mean - expected) < tol


# ---------------------------------------------------------------------------
# Heavy-tailed shape: most users near zero
# ---------------------------------------------------------------------------

def test_sparse_config_majority_below_mean():
    """For sparse attribution, >70% of users should be below the mean rate."""
    pop = sample(SPARSE)
    rates = np.array([u.conversion_rate for u in pop.all_users])
    frac_below_mean = np.mean(rates < SPARSE.expected_rate)
    assert frac_below_mean > 0.70, (
        f"Only {frac_below_mean:.1%} below mean — distribution may not be heavy-tailed enough"
    )


def test_sparse_config_has_high_activity_tail():
    """A non-trivial fraction of users should have rates well above the mean."""
    pop = sample(SPARSE)
    rates = np.array([u.conversion_rate for u in pop.all_users])
    frac_above_3x_mean = np.mean(rates > 3 * SPARSE.expected_rate)
    assert frac_above_3x_mean > 0.05, (
        f"Only {frac_above_3x_mean:.1%} above 3x mean — tail may be too thin"
    )


def test_sparse_has_heavier_tail_than_moderate():
    """Sparse config should concentrate more mass near zero than moderate config."""
    pop_sparse = sample(SPARSE, seed=1)
    pop_moderate = sample(MODERATE, seed=1)
    frac_low_sparse = np.mean(
        [u.conversion_rate for u in pop_sparse.all_users] < np.array(0.05)
    )
    frac_low_moderate = np.mean(
        [u.conversion_rate for u in pop_moderate.all_users] < np.array(0.05)
    )
    assert frac_low_sparse > frac_low_moderate


# ---------------------------------------------------------------------------
# Budget assignment
# ---------------------------------------------------------------------------

def test_epsilon_budget_assigned_correctly():
    config = PopulationConfig(n_users=10, alpha=0.3, beta=3.0,
                              epsilon_budget=2.5, delta_budget=1e-6)
    pop = sample(config)
    assert all(u.epsilon_budget == 2.5 for u in pop.all_users)
    assert all(u.epsilon_spent == 0.0 for u in pop.all_users)


def test_delta_budget_assigned_correctly():
    config = PopulationConfig(n_users=10, alpha=0.3, beta=3.0,
                              epsilon_budget=1.0, delta_budget=1e-5)
    pop = sample(config)
    assert all(u.delta_budget == 1e-5 for u in pop.all_users)
    assert all(u.delta_spent == 0.0 for u in pop.all_users)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_produces_identical_population():
    pop1 = sample(SPARSE, seed=42)
    pop2 = sample(SPARSE, seed=42)
    rates1 = [u.conversion_rate for u in pop1.all_users]
    rates2 = [u.conversion_rate for u in pop2.all_users]
    assert rates1 == rates2


def test_different_seeds_produce_different_populations():
    pop1 = sample(SPARSE, seed=1)
    pop2 = sample(SPARSE, seed=2)
    rates1 = [u.conversion_rate for u in pop1.all_users]
    rates2 = [u.conversion_rate for u in pop2.all_users]
    assert rates1 != rates2


# ---------------------------------------------------------------------------
# describe_population
# ---------------------------------------------------------------------------

def test_describe_population_keys():
    pop = sample(SPARSE)
    stats = describe_population(pop)
    expected_keys = {
        "n_users", "mean", "std", "median",
        "p10", "p25", "p75", "p90", "p99",
        "frac_below_0.05", "frac_below_0.20", "frac_above_0.50",
    }
    assert expected_keys == set(stats.keys())


def test_describe_population_n_users():
    pop = sample(SPARSE)
    stats = describe_population(pop)
    assert stats["n_users"] == SPARSE.n_users


def test_describe_population_percentile_ordering():
    pop = sample(SPARSE)
    stats = describe_population(pop)
    assert stats["p10"] <= stats["p25"] <= stats["median"] <= stats["p75"] <= stats["p90"] <= stats["p99"]


def test_describe_population_fractions_sum_sensibly():
    pop = sample(SPARSE)
    stats = describe_population(pop)
    assert 0.0 <= stats["frac_below_0.05"] <= 1.0
    assert 0.0 <= stats["frac_below_0.20"] <= 1.0
    assert 0.0 <= stats["frac_above_0.50"] <= 1.0
    assert stats["frac_below_0.05"] <= stats["frac_below_0.20"]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_invalid_n_users_raises():
    with pytest.raises(ValueError):
        PopulationConfig(n_users=0, alpha=0.3, beta=3.0,
                         epsilon_budget=1.0, delta_budget=1e-5)


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        PopulationConfig(n_users=100, alpha=0.0, beta=3.0,
                         epsilon_budget=1.0, delta_budget=1e-5)


def test_invalid_beta_raises():
    with pytest.raises(ValueError):
        PopulationConfig(n_users=100, alpha=0.3, beta=-1.0,
                         epsilon_budget=1.0, delta_budget=1e-5)


def test_invalid_epsilon_budget_raises():
    with pytest.raises(ValueError):
        PopulationConfig(n_users=100, alpha=0.3, beta=3.0,
                         epsilon_budget=0.0, delta_budget=1e-5)


def test_negative_delta_budget_raises():
    with pytest.raises(ValueError):
        PopulationConfig(n_users=100, alpha=0.3, beta=3.0,
                         epsilon_budget=1.0, delta_budget=-1.0)
