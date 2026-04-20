import math

import numpy as np
import pytest

from simulation.calibration import q_from_epsilon, set_threshold_for_hit_rate
from simulation.datasets import generate_synthetic_population, PopulationConfig


def make_pop(n=500, alpha=0.3, beta=0.7, seed=0):
    config = PopulationConfig(
        n_users=n, alpha=alpha, beta=beta,
        epsilon_budget=10.0, delta_budget=1.0,
    )
    return generate_synthetic_population(config, rng=np.random.default_rng(seed))


class TestQFromEpsilon:
    def test_lemma_2_3_formula(self):
        for epsilon in [0.1, 0.5, 1.0, 2.0]:
            q = q_from_epsilon(epsilon)
            expected = 1.0 / (math.exp(epsilon) + 1.0)
            assert abs(q - expected) < 1e-12

    def test_larger_epsilon_gives_smaller_q(self):
        assert q_from_epsilon(0.1) > q_from_epsilon(1.0) > q_from_epsilon(5.0)

    def test_approaches_half_for_small_epsilon(self):
        q = q_from_epsilon(1e-6)
        assert abs(q - 0.5) < 1e-5

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError):
            q_from_epsilon(0.0)
        with pytest.raises(ValueError):
            q_from_epsilon(-1.0)


class TestSetThresholdForHitRate:
    def test_empirical_hit_rate_close_to_target(self):
        pop = make_pop()
        T, hit_rate = set_threshold_for_hit_rate(
            pop, epsilon=0.1, target_hit_rate=0.3, n_pilot=5000,
            rng=np.random.default_rng(0),
        )
        assert abs(hit_rate - 0.3) < 0.03, f"hit_rate={hit_rate:.4f} too far from 0.3"

    def test_higher_target_hit_rate_gives_lower_threshold(self):
        pop = make_pop()
        T_low, _ = set_threshold_for_hit_rate(pop, epsilon=0.1, target_hit_rate=0.1, n_pilot=2000, rng=np.random.default_rng(1))
        T_high, _ = set_threshold_for_hit_rate(pop, epsilon=0.1, target_hit_rate=0.5, n_pilot=2000, rng=np.random.default_rng(1))
        assert T_low > T_high, "higher target_hit_rate should require lower threshold"

    def test_larger_epsilon_gives_tighter_threshold(self):
        pop = make_pop()
        T_small, _ = set_threshold_for_hit_rate(pop, epsilon=0.01, target_hit_rate=0.3, n_pilot=2000, rng=np.random.default_rng(2))
        T_large, _ = set_threshold_for_hit_rate(pop, epsilon=10.0, target_hit_rate=0.3, n_pilot=2000, rng=np.random.default_rng(2))
        assert T_small > T_large, "larger noise requires lower threshold to hit the same rate"

    def test_returns_float_threshold(self):
        pop = make_pop()
        T, hit_rate = set_threshold_for_hit_rate(pop, epsilon=0.5, target_hit_rate=0.2, n_pilot=500, rng=np.random.default_rng(3))
        assert isinstance(T, float)
        assert isinstance(hit_rate, float)
        assert 0 < hit_rate < 1

    def test_invalid_target_hit_rate_raises(self):
        pop = make_pop()
        with pytest.raises(ValueError):
            set_threshold_for_hit_rate(pop, epsilon=0.1, target_hit_rate=0.0)
        with pytest.raises(ValueError):
            set_threshold_for_hit_rate(pop, epsilon=0.1, target_hit_rate=1.0)
        with pytest.raises(ValueError):
            set_threshold_for_hit_rate(pop, epsilon=0.1, target_hit_rate=1.5)

    def test_invalid_epsilon_raises(self):
        pop = make_pop()
        with pytest.raises(ValueError):
            set_threshold_for_hit_rate(pop, epsilon=0.0, target_hit_rate=0.3)
        with pytest.raises(ValueError):
            set_threshold_for_hit_rate(pop, epsilon=-1.0, target_hit_rate=0.3)

    def test_reproducible_with_same_seed(self):
        pop = make_pop()
        T1, r1 = set_threshold_for_hit_rate(pop, epsilon=0.1, target_hit_rate=0.3, n_pilot=500, rng=np.random.default_rng(42))
        T2, r2 = set_threshold_for_hit_rate(pop, epsilon=0.1, target_hit_rate=0.3, n_pilot=500, rng=np.random.default_rng(42))
        assert T1 == T2 and r1 == r2

    def test_hit_rate_much_less_than_q(self):
        """Verify the recommended usage: target_hit_rate << q_from_epsilon(epsilon)."""
        pop = make_pop()
        epsilon = 0.1
        q = q_from_epsilon(epsilon)
        target = 0.05
        T, hit_rate = set_threshold_for_hit_rate(pop, epsilon=epsilon, target_hit_rate=target, n_pilot=3000, rng=np.random.default_rng(7))
        assert hit_rate < q, f"hit_rate={hit_rate:.3f} should be << q={q:.3f}"
        assert abs(hit_rate - target) < 0.03
