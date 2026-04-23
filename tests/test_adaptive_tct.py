"""
Tests for AdaptiveTCTAccountant.

Organised into:
  1. Mathematical primitives (alpha, compositions)
  2. Registration (RegisterForTCT)
  3. Hit charging (ChargeTargetHit)
  4. Full charge() logic (combining registration + hit charging)
  5. Telescoping / budget correctness
  6. Comparison against StandardDPAccountant (TCT cheaper when empirical hit rate << q)
  7. Input validation
"""

import math

import numpy as np
import pytest

from simulation.accountants.adaptive_tct import AdaptiveTCTAccountant
from simulation.accountants.standard_dp import StandardDPAccountant
from simulation.calibration import q_from_epsilon
from simulation.datasets import PopulationConfig, generate_synthetic_population
from simulation.environment import SimulationEnvironment
from simulation.population import User, UserPopulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_user(uid=0, eps_budget=1000.0, delta_budget=1.0) -> User:
    return User(user_id=uid, epsilon_budget=eps_budget,
                delta_budget=delta_budget, conversion_rate=0.5)


def make_tct(epsilon=0.1, delta=1e-5) -> AdaptiveTCTAccountant:
    return AdaptiveTCTAccountant(epsilon=epsilon, delta=delta)


# ---------------------------------------------------------------------------
# 1. Mathematical primitives
# ---------------------------------------------------------------------------

class TestAlpha:
    def test_alpha_positive(self):
        tct = make_tct(delta=1e-5)
        for i in range(1, 20):
            assert tct.alpha(i) > 0, f"alpha({i}) should be positive"

    def test_alpha_decreasing(self):
        """Remark 4: alpha_i decreases as i grows."""
        tct = make_tct(delta=1e-5)
        alphas = [tct.alpha(i) for i in range(1, 50)]
        assert all(a > b for a, b in zip(alphas, alphas[1:])), \
            "alpha should be strictly decreasing"

    def test_alpha_satisfies_quadratic(self):
        """alpha_i is the positive root of i*alpha^2 - 2*beta_i*alpha - 2*beta_i = 0."""
        tct = make_tct(delta=1e-3)
        for i in range(1, 10):
            a = tct.alpha(i)
            delta_i = (6.0 / (math.pi**2 * i**2)) * tct.delta
            beta_i = math.log(1.0 / delta_i)
            residual = i * a**2 - 2 * beta_i * a - 2 * beta_i
            assert abs(residual) < 1e-8, \
                f"alpha({i}) = {a} does not satisfy quadratic (residual={residual})"

    def test_alpha_approaches_zero_for_large_i(self):
        """For large i, alpha_i -> 0 (Remark 4 asymptotic)."""
        tct = make_tct(delta=1e-2)
        assert tct.alpha(10000) < 0.1

    def test_alpha_invalid_i_raises(self):
        tct = make_tct()
        with pytest.raises(ValueError):
            tct.alpha(0)


class TestCompositions:
    def test_compositions_zero_at_k0(self):
        assert make_tct().compositions(0) == 0.0

    def test_compositions_positive_for_k_ge_1(self):
        tct = make_tct()
        for k in range(1, 15):
            assert tct.compositions(k) > 0

    def test_compositions_increasing(self):
        """K_k should be strictly increasing in k."""
        tct = make_tct()
        ks = [tct.compositions(k) for k in range(0, 20)]
        assert all(a < b for a, b in zip(ks, ks[1:])), "K_k should be increasing"

    def test_compositions_formula(self):
        """K_k = (1 + alpha_k) / q * k, where q = 1/(e^epsilon + 1)."""
        tct = make_tct(delta=1e-4)
        for k in range(1, 10):
            expected = (1.0 + tct.alpha(k)) / tct.q * k
            assert abs(tct.compositions(k) - expected) < 1e-10

    def test_compositions_invalid_k_raises(self):
        with pytest.raises(ValueError):
            make_tct().compositions(-1)

    def test_q_equals_lemma_2_3_value(self):
        """q must equal 1/(e^epsilon + 1) from Lemma 2.3 (NotPrior target)."""
        for eps in [0.01, 0.1, 0.5, 1.0, 2.0]:
            tct = AdaptiveTCTAccountant(epsilon=eps, delta=1e-3)
            expected_q = 1.0 / (math.exp(eps) + 1.0)
            assert abs(tct.q - expected_q) < 1e-12, \
                f"epsilon={eps}: q={tct.q}, expected {expected_q}"

    def test_larger_epsilon_gives_smaller_q(self):
        """Larger epsilon → smaller q = 1/(e^ε+1) → more compositions needed."""
        tct_lo = AdaptiveTCTAccountant(epsilon=0.1, delta=1e-3)  # q ≈ 0.475
        tct_hi = AdaptiveTCTAccountant(epsilon=1.0, delta=1e-3)  # q ≈ 0.269
        assert tct_lo.q > tct_hi.q
        for k in range(1, 10):
            assert tct_lo.compositions(k) < tct_hi.compositions(k), \
                f"k={k}: smaller epsilon should give fewer compositions"

    def test_smaller_delta_gives_more_compositions(self):
        """Smaller delta -> smaller delta_i -> larger alpha -> more compositions."""
        tct_lo = make_tct(delta=1e-8)
        tct_hi = make_tct(delta=1e-2)
        for k in range(1, 10):
            assert tct_lo.compositions(k) > tct_hi.compositions(k)


# ---------------------------------------------------------------------------
# 2. Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_registration_charges_correct_epsilon(self):
        """RegisterForTCT should charge K_1 * epsilon."""
        tct = make_tct(epsilon=0.1, delta=1e-5)
        user = make_user()
        eps_cost, _ = tct.charge(user, is_top=False, contributed=False, round_num=0)
        expected = tct.compositions(1) * tct.epsilon
        assert abs(eps_cost - expected) < 1e-10

    def test_registration_charges_full_delta(self):
        """RegisterForTCT should charge the full system delta."""
        tct = make_tct(epsilon=0.1, delta=1e-5)
        user = make_user()
        _, delta_cost = tct.charge(user, is_top=False, contributed=False, round_num=0)
        assert abs(delta_cost - tct.delta) < 1e-15

    def test_registration_sets_cx_to_1(self):
        tct = make_tct()
        user = make_user()
        assert tct.hit_count(user.user_id) == 0
        tct.charge(user, is_top=False, contributed=False, round_num=0)
        assert tct.hit_count(user.user_id) == 1

    def test_registration_happens_only_once(self):
        """Subsequent non-top rounds should cost 0 after registration."""
        tct = make_tct()
        user = make_user()
        first_eps, first_delta = tct.charge(user, is_top=False, contributed=False, round_num=0)
        assert first_eps > 0 and first_delta > 0

        for r in range(1, 5):
            eps, delta = tct.charge(user, is_top=False, contributed=False, round_num=r)
            assert eps == 0.0, f"Round {r}: expected 0 eps cost, got {eps}"
            assert delta == 0.0, f"Round {r}: expected 0 delta cost, got {delta}"


# ---------------------------------------------------------------------------
# 3. Hit charging
# ---------------------------------------------------------------------------

class TestHitCharging:
    def test_no_charge_on_non_top_round(self):
        tct = make_tct()
        user = make_user()
        tct.charge(user, is_top=False, contributed=True, round_num=0)  # register
        eps, delta = tct.charge(user, is_top=False, contributed=True, round_num=1)
        assert eps == 0.0
        assert delta == 0.0

    def test_no_charge_on_top_round_without_contribution(self):
        tct = make_tct()
        user = make_user()
        tct.charge(user, is_top=False, contributed=False, round_num=0)  # register
        eps, delta = tct.charge(user, is_top=True, contributed=False, round_num=1)
        assert eps == 0.0
        assert delta == 0.0

    def test_hit_charge_increments_cx(self):
        tct = make_tct()
        user = make_user()
        tct.charge(user, is_top=False, contributed=False, round_num=0)
        assert tct.hit_count(user.user_id) == 1
        tct.charge(user, is_top=True, contributed=True, round_num=1)
        assert tct.hit_count(user.user_id) == 2
        tct.charge(user, is_top=True, contributed=True, round_num=2)
        assert tct.hit_count(user.user_id) == 3

    def test_hit_charge_delta_is_zero(self):
        tct = make_tct()
        user = make_user()
        tct.charge(user, is_top=False, contributed=False, round_num=0)
        for r in range(1, 6):
            _, delta_cost = tct.charge(user, is_top=True, contributed=True, round_num=r)
            assert delta_cost == 0.0

    def test_hit_charge_epsilon_is_correct(self):
        """ChargeTargetHit should charge (K_{cx+1} - K_cx) * epsilon."""
        tct = make_tct(epsilon=0.1, delta=1e-5)
        user = make_user()
        tct.charge(user, is_top=False, contributed=False, round_num=0)
        cx_before = tct.hit_count(user.user_id)

        eps_cost, _ = tct.charge(user, is_top=True, contributed=True, round_num=1)
        expected = (tct.compositions(cx_before + 1) - tct.compositions(cx_before)) * tct.epsilon
        assert abs(eps_cost - expected) < 1e-10


# ---------------------------------------------------------------------------
# 4. Combined charge() logic
# ---------------------------------------------------------------------------

class TestCombinedCharge:
    def test_first_round_top_with_contribution(self):
        """On first round, if it's a top and user contributed, both register and hit charge apply."""
        tct = make_tct(epsilon=0.1, delta=1e-5)
        user = make_user()
        eps_cost, delta_cost = tct.charge(user, is_top=True, contributed=True, round_num=0)

        assert abs(eps_cost - tct.compositions(2) * tct.epsilon) < 1e-10
        assert abs(delta_cost - tct.delta) < 1e-15
        assert tct.hit_count(user.user_id) == 2

    def test_independent_users_have_independent_state(self):
        tct = make_tct()
        u1 = make_user(uid=0)
        u2 = make_user(uid=1)

        tct.charge(u1, is_top=True, contributed=True, round_num=0)
        tct.charge(u2, is_top=False, contributed=False, round_num=0)

        assert tct.hit_count(0) == 2
        assert tct.hit_count(1) == 1


# ---------------------------------------------------------------------------
# 5. Telescoping / budget correctness
# ---------------------------------------------------------------------------

class TestTelescoping:
    def _run_k_hits(self, k: int, epsilon=0.1, delta=1e-4) -> tuple[float, float]:
        """
        Fresh accountant + fresh user. Simulates k actual target hits
        preceded by a registration-only round. Returns total (eps, delta) charged.
        """
        tct = AdaptiveTCTAccountant(epsilon=epsilon, delta=delta)
        user = make_user(uid=0, eps_budget=1e9, delta_budget=1.0)
        total_eps = 0.0
        total_delta = 0.0

        e, d = tct.charge(user, is_top=False, contributed=False, round_num=0)
        total_eps += e
        total_delta += d

        for r in range(1, k + 1):
            e, d = tct.charge(user, is_top=True, contributed=True, round_num=r)
            total_eps += e
            total_delta += d

        return total_eps, total_delta

    def test_total_eps_after_k_hits_equals_K_kplus1_times_epsilon(self):
        """
        After k actual target hits, total epsilon charged = K_{k+1} * epsilon.
        This is the telescoping property required by the Theorem 5 proof.
        """
        for k in range(0, 8):
            total_eps, _ = self._run_k_hits(k, epsilon=0.1, delta=1e-4)
            tct = AdaptiveTCTAccountant(epsilon=0.1, delta=1e-4)
            expected = tct.compositions(k + 1) * tct.epsilon
            assert abs(total_eps - expected) < 1e-8, \
                f"k={k}: total_eps={total_eps:.6f}, expected K_{{k+1}}*eps={expected:.6f}"

    def test_total_delta_is_exactly_system_delta(self):
        """Delta is fully pre-paid at registration and never charged again."""
        for k in range(0, 8):
            _, total_delta = self._run_k_hits(k, epsilon=0.05, delta=1e-5)
            assert abs(total_delta - 1e-5) < 1e-15, \
                f"k={k}: total_delta should equal system delta 1e-5"

    def test_incremental_charges_are_positive(self):
        """Each incremental hit charge should be strictly positive (K is increasing)."""
        tct = make_tct(epsilon=0.1, delta=1e-5)
        user = make_user(uid=0, eps_budget=1e9, delta_budget=1.0)
        tct.charge(user, is_top=False, contributed=False, round_num=0)
        for r in range(1, 10):
            eps_cost, _ = tct.charge(user, is_top=True, contributed=True, round_num=r)
            assert eps_cost > 0, f"Round {r}: incremental eps charge should be positive"


# ---------------------------------------------------------------------------
# 6. TCT cheaper than standard IDP when empirical hit rate << q
# ---------------------------------------------------------------------------

class TestTCTAdvantage:
    """
    TCT's advantage over standard IDP is visible when:
      (a) enough rounds have passed that IDP's cumulative cost exceeds TCT's
          registration cost (break-even ≈ K_1 rounds), AND
      (b) the empirical hit rate is much less than q = 1/(e^ε+1).

    With epsilon=0.1: q = 1/(e^0.1+1) ≈ 0.475, K_1 ≈ 35 rounds to break even.
    We use threshold=1e6 (no hits ever) and n_rounds=300 >> 35 break-even.
    Registration cost = K_1 * epsilon ≈ 3.5 epsilon.
    """

    EPSILON = 0.1
    DELTA = 1e-3
    N_ROUNDS = 300
    THRESHOLD = 1e6  # no tops ever — isolates registration vs per-round cost

    def _simulate(self, accountant, eps_budget, delta_budget=1.0,
                  n_users=100, seed=0):
        config = PopulationConfig(
            n_users=n_users, alpha=0.3, beta=3.0,
            epsilon_budget=eps_budget, delta_budget=delta_budget,
        )
        pop = generate_synthetic_population(config, np.random.default_rng(seed))
        env = SimulationEnvironment(
            population=pop,
            accountant=accountant,
            threshold=self.THRESHOLD,
            query_epsilon=self.EPSILON,
            num_rounds=self.N_ROUNDS,
            rng=np.random.default_rng(seed),
        )
        return env.run()

    def test_tct_users_survive_when_idp_users_deplete(self):
        """
        With eps_budget = N_ROUNDS * epsilon (enough for IDP to exhaust),
        TCT users survive because registration (~3.5 eps) << total IDP cost (30 eps).
        """
        eps_budget = self.N_ROUNDS * self.EPSILON  # = 30.0

        idp = StandardDPAccountant(epsilon_per_round=self.EPSILON)
        tct = AdaptiveTCTAccountant(epsilon=self.EPSILON, delta=self.DELTA)

        r_idp = self._simulate(idp, eps_budget=eps_budget)
        r_tct = self._simulate(tct, eps_budget=eps_budget)

        assert r_idp.final_active_fraction < 0.05
        assert r_tct.final_active_fraction > 0.95

    def test_tct_higher_average_budget_remaining(self):
        """
        With ample budget (no depletion), TCT's average remaining epsilon should
        exceed IDP's after N_ROUNDS, because TCT charges only registration while
        IDP charges every round.
        """
        eps_budget = 200.0

        idp = StandardDPAccountant(epsilon_per_round=self.EPSILON)
        tct = AdaptiveTCTAccountant(epsilon=self.EPSILON, delta=self.DELTA)

        r_idp = self._simulate(idp, eps_budget=eps_budget)
        r_tct = self._simulate(tct, eps_budget=eps_budget)

        idp_final = np.mean(r_idp.avg_epsilon_remaining_by_round()[-1:])
        tct_final = np.mean(r_tct.avg_epsilon_remaining_by_round()[-1:])

        assert tct_final > idp_final, (
            f"TCT remaining {tct_final:.2f} should exceed IDP remaining {idp_final:.2f}"
        )


# ---------------------------------------------------------------------------
# 7. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError):
            AdaptiveTCTAccountant(epsilon=-0.1, delta=1e-5)

    def test_zero_epsilon_raises(self):
        with pytest.raises(ValueError):
            AdaptiveTCTAccountant(epsilon=0.0, delta=1e-5)

    def test_delta_zero_raises(self):
        with pytest.raises(ValueError):
            AdaptiveTCTAccountant(epsilon=0.1, delta=0.0)

    def test_delta_one_raises(self):
        with pytest.raises(ValueError):
            AdaptiveTCTAccountant(epsilon=0.1, delta=1.0)
