import numpy as np
import pytest

from simulation.calibration import calibrate_threshold
from simulation.datasets import generate_synthetic_population, PopulationConfig


def make_pop(n=500, alpha=0.3, beta=0.7, seed=0):
    config = PopulationConfig(
        n_users=n, alpha=alpha, beta=beta,
        epsilon_budget=10.0, delta_budget=1.0,
    )
    return generate_synthetic_population(config, rng=np.random.default_rng(seed))


class TestCalibrateThreshold:
    def test_empirical_hit_rate_close_to_target(self):
        pop = make_pop()
        T, q_emp = calibrate_threshold(
            pop, epsilon=0.1, target_q=0.3, n_pilot=5000,
            rng=np.random.default_rng(0),
        )
        # Empirical rate should be within ±3% of target
        assert abs(q_emp - 0.3) < 0.03, f"q_emp={q_emp:.4f} too far from 0.3"

    def test_higher_target_q_gives_lower_threshold(self):
        pop = make_pop()
        T_low, _ = calibrate_threshold(pop, epsilon=0.1, target_q=0.1, n_pilot=2000, rng=np.random.default_rng(1))
        T_high, _ = calibrate_threshold(pop, epsilon=0.1, target_q=0.5, n_pilot=2000, rng=np.random.default_rng(1))
        assert T_low > T_high, "higher target_q should require lower threshold"

    def test_larger_epsilon_gives_tighter_threshold(self):
        # With larger ε the noise is smaller, so threshold is set closer to
        # the true-count quantile. The noisy and noiseless quantiles converge.
        pop = make_pop()
        T_small, _ = calibrate_threshold(pop, epsilon=0.01, target_q=0.3, n_pilot=2000, rng=np.random.default_rng(2))
        T_large, _ = calibrate_threshold(pop, epsilon=10.0, target_q=0.3, n_pilot=2000, rng=np.random.default_rng(2))
        # With tiny ε (huge noise), threshold should be much lower to compensate
        assert T_small > T_large, "larger noise requires lower threshold to hit the same q"

    def test_returns_float_threshold(self):
        pop = make_pop()
        T, q_emp = calibrate_threshold(pop, epsilon=0.5, target_q=0.2, n_pilot=500, rng=np.random.default_rng(3))
        assert isinstance(T, float)
        assert isinstance(q_emp, float)
        assert 0 < q_emp < 1

    def test_invalid_target_q_raises(self):
        pop = make_pop()
        with pytest.raises(ValueError):
            calibrate_threshold(pop, epsilon=0.1, target_q=0.0)
        with pytest.raises(ValueError):
            calibrate_threshold(pop, epsilon=0.1, target_q=1.0)
        with pytest.raises(ValueError):
            calibrate_threshold(pop, epsilon=0.1, target_q=1.5)

    def test_invalid_epsilon_raises(self):
        pop = make_pop()
        with pytest.raises(ValueError):
            calibrate_threshold(pop, epsilon=0.0, target_q=0.3)
        with pytest.raises(ValueError):
            calibrate_threshold(pop, epsilon=-1.0, target_q=0.3)

    def test_reproducible_with_same_seed(self):
        pop = make_pop()
        T1, q1 = calibrate_threshold(pop, epsilon=0.1, target_q=0.3, n_pilot=500, rng=np.random.default_rng(42))
        T2, q2 = calibrate_threshold(pop, epsilon=0.1, target_q=0.3, n_pilot=500, rng=np.random.default_rng(42))
        assert T1 == T2 and q1 == q2
