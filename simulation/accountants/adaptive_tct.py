"""
Adaptive TCT accountant — implements Algorithms 3–5 and Corollaries 2/4 (paper).

Correspondence to paper:
  alpha(i)               — Proposition 4 dynamic alpha (Zeta-2 allocation)
  compositions(k)        — K_k, prepaid composition count (dispatches by schedule)
  compositions_zeta2(k)  — Proposition 4: T_k = (1+η_k)·k/q
  compositions_stitched(k) — Proposition 5: T_k = k/q + max(8β_j/q, √(c'(1-q)kβ_j)/q)
  budget_cost(K)         — cost charged for K compositions (dispatches by mode)
  _register()            — RegisterForTCT (Algorithm 3)
  _charge_hit()          — ChargeTargetHit (Algorithm 4)
  charge()               — IDP filter logic (Algorithm 5)

===========================================================================
Composition accountant choice
===========================================================================

Simple ε-DP composition (alpha_rdp=None, delta_prime=None, default):
  charge = K_k * epsilon
  delta_charge_at_registration = delta (= delta_infinity)

  Gives (K_k * epsilon, delta_infinity)-DP on the good event.
  Total IDP guarantee: (epsilon_init, delta_infinity)-IDP.

Advanced composition (delta_prime > 0, Rogers et al. 2016):
  charge = epsilon_AC(K_k, epsilon, delta_prime)
  delta_charge_at_registration = delta_infinity + delta_prime

  Gives (epsilon_init, delta_init)-IDP, using the full delta budget.
  Benefit: O(sqrt(K)) vs O(K) epsilon growth for heavy users.

RDP composition (alpha_rdp > 1, Mironov 2017 / Corollary 2 + Corollary 4):
  rho = alpha_rdp * epsilon**2 / 2   (ε-DP → RDP conversion bound)
  charge = K_k * rho                 (Eq. 8 / Corollary 4: phi_alpha(t))
  delta_charge_at_registration = delta (= delta_infinity)

  Gives (alpha_rdp, K_k * rho, delta_infinity)-RDP on the good event.
  Total IDP guarantee: (alpha_rdp, epsilon_rdp_init, delta_infinity)-approx-RDP
  (Definition 2 / Theorem 4).  Budget is interpreted as epsilon_rdp_init.

  Benefit: rho = alpha * epsilon^2 / 2 << epsilon for small epsilon,
  so the per-composition cost is O(epsilon^2) rather than O(epsilon).

===========================================================================
Anytime schedule choice (schedule parameter)
===========================================================================

'zeta2' (default) — Proposition 4 (Nonuniform-budget union bound):
  δ_k = 6/(π²k²)·δ,  β_k = log(1/δ_k)
  η_k = (β_k + √(β_k² + 2β_k·k)) / k
  K_k = (1 + η_k)·k/q
  Overhead: Θ(√(k·log(k/δ))/q)

'stitched' — Proposition 5 (Stitched anytime concentration bound):
  j(k) = ⌈log₂(k ∨ 2)⌉,  δ_j = 6/(π²j²)·δ,  β_j = log(π²j²/(6δ))
  K_k = k/q + max(8β_j/q, √(c'·(1-q)·k·β_j)/q)
  c' = (73√73 − 379)·(1+√2)² / 216
  Overhead: Θ(√((1-q)·k·log(log k))/q)  — asymptotically tighter than Zeta-2

The 'stitched' schedule is sharper for large k because it unions over epochs
(j ~ log log k) rather than individual hit counts (β_k ~ log k).

===========================================================================
Note on q (Lemma 2.3 of Cohen & Lyu 2023):
  For the Laplace threshold mechanism with bottom = FALSE (below threshold),
  the NotPrior target top = {TRUE} is always a 1/(e^epsilon + 1)-target for
  any epsilon-DP algorithm — regardless of the threshold T or the data
  distribution. Therefore q is determined solely by epsilon:

      q = 1 / (e^epsilon + 1)

  This is NOT the empirical hit rate (fraction of rounds that cross T).
  The threshold T is set independently based on application requirements.
  TCT is beneficial when most rounds output bottom (empirical hit rate << q).
"""

from __future__ import annotations

import math
from typing import Literal, Optional

from simulation.population import User
from .base import PrivacyAccountant

# Proposition 5 constant: c' = (73√73 − 379)·(1+√2)² / 216
_C_PRIME = (73 * math.sqrt(73) - 379) * (1 + math.sqrt(2)) ** 2 / 216


class AdaptiveTCTAccountant(PrivacyAccountant):
    """
    Adaptive TCT accountant with IDP filters (Algorithms 3–5, Corollaries 2/4).

    q is derived from epsilon via the NotPrior target (Cohen & Lyu 2023):
        q = 1 / (e^epsilon + 1)

    Charging structure per user:
      - First participation (cx = 0): RegisterForTCT charges
        budget_cost(K_1) from the budget and delta_charge from delta. Sets cx = 1.
      - Each target hit the user contributed to: ChargeTargetHit charges
        budget_cost(K_{cx+1}) - budget_cost(K_cx). No further delta.
      - Non-top rounds (or top rounds where user didn't contribute): zero.

    The telescoping property ensures that after k hits the filter has charged
    exactly budget_cost(K_{k+1}) total.

    Args:
        epsilon:     per-query ε for the Laplace mechanism
        delta:       TCT anytime failure probability δ∞, prepaid at registration
        delta_prime: if not None, use advanced (ε,δ)-DP composition (Rogers et al.
                     2016). Mutually exclusive with alpha_rdp.
        alpha_rdp:   if not None, use RDP composition at this Rényi order α > 1
                     (Corollary 2 / Corollary 4). Per-call cost ρ = α·ε²/2.
                     Budget is interpreted as ε_RDP. Mutually exclusive with
                     delta_prime.
        independent: if True (default), ChargeTargetHit fires only for users
                     who personally contributed to the hit.
        schedule:    anytime composition schedule.
                     'zeta2'    — Proposition 4 (default): overhead Θ(√(k log k)/q)
                     'stitched' — Proposition 5: overhead Θ(√((1-q)k log log k)/q)
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        delta_prime: Optional[float] = None,
        alpha_rdp: Optional[float] = None,
        independent: bool = True,
        schedule: Literal["zeta2", "stitched"] = "zeta2",
    ):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if delta_prime is not None and alpha_rdp is not None:
            raise ValueError("delta_prime and alpha_rdp are mutually exclusive")
        if delta_prime is not None:
            if not 0 < delta_prime < 1:
                raise ValueError(f"delta_prime must be in (0, 1), got {delta_prime}")
            if delta + delta_prime >= 1:
                raise ValueError(
                    f"delta + delta_prime must be < 1, got {delta + delta_prime}"
                )
        if alpha_rdp is not None and alpha_rdp <= 1:
            raise ValueError(f"alpha_rdp (Rényi order) must be > 1, got {alpha_rdp}")
        if schedule not in ("zeta2", "stitched"):
            raise ValueError(f"schedule must be 'zeta2' or 'stitched', got {schedule!r}")

        self.epsilon = epsilon
        self.delta = delta
        self.delta_prime = delta_prime
        self.alpha_rdp = alpha_rdp
        self.independent = independent
        self.schedule = schedule
        self.q = 1.0 / (math.exp(epsilon) + 1.0)

        self._cx: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Core mathematical primitives
    # ------------------------------------------------------------------

    def alpha(self, i: int) -> float:
        """
        Dynamic alpha for step i (Proposition 4, Zeta-2 allocation).

        delta_k = 6 / (pi^2 * k^2) * delta
        beta_k  = ln(1 / delta_k)
        alpha_k = (beta_k + sqrt(beta_k^2 + 2 * beta_k * k)) / k
        """
        if i < 1:
            raise ValueError(f"i must be >= 1, got {i}")
        delta_i = (6.0 / (math.pi ** 2 * i ** 2)) * self.delta
        beta_i = math.log(1.0 / delta_i)
        discriminant = beta_i ** 2 + 2.0 * beta_i * i
        return (beta_i + math.sqrt(discriminant)) / i

    def compositions_zeta2(self, k: int) -> float:
        """
        Proposition 4 (Nonuniform-budget / Zeta-2) schedule.

        K_k = (1 + eta_k) * k / q,  where eta_k = alpha(k).
        Overhead: Theta(sqrt(k * log(k/delta)) / q).
        K_0 = 0 by convention.
        """
        if k == 0:
            return 0.0
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        return (1.0 + self.alpha(k)) / self.q * k

    def compositions_stitched(self, k: int) -> float:
        """
        Proposition 5 (Stitched anytime concentration bound) schedule.

        j(k) = ceil(log2(k v 2))
        delta_j = 6 / (pi^2 * j^2) * delta
        beta_j  = log(pi^2 * j^2 / (6 * delta))
        K_k = k/q + max(8*beta_j/q, sqrt(c'*(1-q)*k*beta_j)/q)

        where c' = (73*sqrt(73) - 379) * (1+sqrt(2))^2 / 216.
        Overhead: Theta(sqrt((1-q)*k*log(log k)) / q) — tighter than Zeta-2.
        K_0 = 0 by convention.
        """
        if k == 0:
            return 0.0
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        j = math.ceil(math.log2(max(k, 2)))
        beta_j = math.log(math.pi ** 2 * j ** 2 / (6.0 * self.delta))
        additive = 8.0 * beta_j / self.q
        diffusive = math.sqrt(_C_PRIME * (1.0 - self.q) * k * beta_j) / self.q
        return k / self.q + max(additive, diffusive)

    def compositions(self, k: int) -> float:
        """
        Prepaid composition count K_k under self.schedule ('zeta2' or 'stitched').
        K_0 = 0 by convention.
        """
        if self.schedule == "stitched":
            return self.compositions_stitched(k)
        return self.compositions_zeta2(k)

    @staticmethod
    def epsilon_advanced_composition(
        K: float, epsilon: float, delta_prime: float
    ) -> float:
        """
        Advanced composition epsilon for K rounds of epsilon-DP mechanisms.

        From Dwork, Rothblum & Vadhan (2010) / Rogers et al. (2016):

            epsilon_AC(K, epsilon, delta') =
                epsilon * sqrt(2 * K * ln(1/delta'))
                + K * epsilon * (exp(epsilon) - 1)

        Valid for any adaptive stopping time at most K (Rogers et al. 2016,
        Theorem 3.1 pay-as-you-go composition). Monotone increasing in K,
        so applying it with the prepaid bound K_k gives a valid upper bound
        on the actual privacy loss when the number of compositions is random
        and bounded by K_k.

        Grows as O(sqrt(K) * ln(1/delta') * epsilon) vs O(K * epsilon) for
        simple composition — significant saving for large K (many hits).
        """
        if K <= 0:
            return 0.0
        return (
            epsilon * math.sqrt(2.0 * K * math.log(1.0 / delta_prime))
            + K * epsilon * (math.exp(epsilon) - 1.0)
        )

    def budget_cost(self, K: float) -> float:
        """
        Total budget cost for K prepaid compositions.

        Simple ε-DP  (alpha_rdp=None, delta_prime=None):  K * epsilon
        Advanced comp (delta_prime set):  epsilon_AC(K, epsilon, delta_prime)
        RDP           (alpha_rdp set):    K * alpha_rdp * epsilon**2 / 2
                                          (Corollary 4, Eq. 8: K * rho)
        """
        if self.alpha_rdp is not None:
            return K * self.alpha_rdp * self.epsilon ** 2 / 2.0
        if self.delta_prime is None:
            return K * self.epsilon
        return self.epsilon_advanced_composition(K, self.epsilon, self.delta_prime)

    # keep old name as alias for backward compatibility
    def epsilon_budget(self, K: float) -> float:
        return self.budget_cost(K)

    # ------------------------------------------------------------------
    # IDP filter operations (Algorithms 2 and 3)
    # ------------------------------------------------------------------

    def _register(self, user: User) -> tuple[float, float]:
        """
        RegisterForTCT (Algorithm 3): one-time registration charge.

        ε-DP simple:   charges K_1·ε, δ∞ from delta.
        Advanced comp: charges εAC(K_1, ε, δ'), (δ∞ + δ') from delta.
        RDP:           charges K_1·ρ (Corollary 4 initial reservation), δ∞ from delta.

        Sets cx = 1.
        """
        k1_cost = self.budget_cost(self.compositions(1))
        delta_charge = self.delta + (self.delta_prime or 0.0)
        self._cx[user.user_id] = 1
        return (k1_cost, delta_charge)

    def _charge_hit(self, user: User) -> tuple[float, float]:
        """
        ChargeTargetHit (Algorithm 4): incremental charge on each hit.

        Extends prepaid budget from budget_cost(K_cx) to budget_cost(K_{cx+1}).
        For RDP: increment is ϕ_α(cx+1) = (K_{cx+1} - K_cx) · ρ (Eq. 8).
        The increment is always non-negative because budget_cost is monotone in K.

        Delta cost is 0 — fully pre-paid at registration.
        """
        cx = self._cx[user.user_id]
        cost_next = self.budget_cost(self.compositions(cx + 1))
        cost_curr = self.budget_cost(self.compositions(cx))
        self._cx[user.user_id] = cx + 1
        return (cost_next - cost_curr, 0.0)

    # ------------------------------------------------------------------
    # PrivacyAccountant interface
    # ------------------------------------------------------------------

    def charge(
        self,
        user: User,
        is_top: bool,
        contributed: bool,
        round_num: int,
    ) -> tuple[float, float]:
        """
        IDP filter charge for one round (Algorithm 5).

        1. RegisterForTCT if cx == 0 (pre-pays budget_cost(K_1) + delta charge).
        2. ChargeTargetHit if is_top AND user contributed (independent mode)
           or is_top AND registered (non-independent mode).

        Both can fire in the same round. Non-top rounds cost zero after
        registration — the key efficiency advantage of TCT over standard DP.
        """
        eps_cost = 0.0
        delta_cost = 0.0

        if self._cx.get(user.user_id, 0) == 0:
            r_eps, r_delta = self._register(user)
            eps_cost += r_eps
            delta_cost += r_delta

        if is_top and (contributed or not self.independent):
            h_eps, h_delta = self._charge_hit(user)
            eps_cost += h_eps
            delta_cost += h_delta

        return (eps_cost, delta_cost)

    # ------------------------------------------------------------------
    # Analysis helpers (used in H2 experiments)
    # ------------------------------------------------------------------

    def hit_count(self, user_id: int) -> int:
        return self._cx.get(user_id, 0)

    def epsilon_prepaid(self, user_id: int) -> float:
        return self.budget_cost(self.compositions(self.hit_count(user_id)))

    def theoretical_bound(self, k_hits: int) -> float:
        return self.compositions(k_hits)
