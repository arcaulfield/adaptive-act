"""
Adaptive TCT accountant — implements Algorithms 4, 5, 7, 8 and Theorem 3.

Correspondence to paper:
  alpha(i)        — Theorem 3 dynamic alpha (Zeta-2 allocation)
  compositions(k) — K_k, the prepaid composition count for k hits (Algs 7/8)
  _register()     — RegisterForTCT (Algorithms 4/7)
  _charge_hit()   — ChargeTargetHit (Algorithms 5/8)
  charge()        — IDP filter logic from Algorithm 6 (lines 4–12)

Privacy cost model: pure DP linear composition. Each composition of the
epsilon-DP Laplace mechanism costs epsilon in privacy, so the total cost for
K_k compositions is K_k * epsilon. Algorithm 8's `∆cost` is therefore
(K_i - K_{i-1}) * epsilon.

Note on q (Lemma 2.3 of Cohen & Lyu 2023):
  For the Laplace threshold mechanism with ⊥ = FALSE (below threshold),
  the NotPrior target ⊤ = {TRUE} is always a 1/(e^ε + 1)-target for any
  ε-DP algorithm — regardless of the threshold T or the data distribution.
  Therefore q is determined solely by epsilon:

      q = 1 / (e^ε + 1)

  This is NOT the empirical hit rate (fraction of rounds that cross T).
  The threshold T is set independently based on application requirements.
  TCT is beneficial when most rounds output ⊥ (empirical hit rate << q).

Note on Algorithm 5 vs Algorithm 8: Algorithm 5 gives a simplified formula
R = (alpha_i - alpha_{i-1}) / (Fx.cx / q) * epsilon. Because alpha is
decreasing, this yields negative charges — a bug in the paper. Algorithm 8
computes the increment as (K_i - K_{i-1}) * epsilon, which telescopes
correctly to C_i = K_i * epsilon as required by the Theorem 5 proof.
"""

import math

from simulation.population import User
from .base import PrivacyAccountant


class AdaptiveTCTAccountant(PrivacyAccountant):
    """
    Adaptive TCT accountant with IDP filters (Algorithms 4–8, Theorem 3).

    q is derived from epsilon via Lemma 2.3 (NotPrior target):
        q = 1 / (e^epsilon + 1)
    This is a universal lower bound on the hit probability in the private
    branch of any epsilon-DP mechanism, independent of threshold or data.

    Charging structure per user:
      - First participation (cx = 0): RegisterForTCT charges K_1 * epsilon
        from epsilon budget and the full system delta from delta budget.
      - Each target hit the user contributed to: ChargeTargetHit charges
        (K_{cx+1} - K_cx) * epsilon; no additional delta.
      - Non-top rounds (or top rounds where the user didn't contribute):
        zero charge.

    The telescoping property ensures that after k actual target hits the
    filter has charged exactly C_{k+1} = K_{k+1} * epsilon, which upper
    bounds the actual privacy loss K_k * epsilon with high probability.

    Args:
        epsilon:    per-query privacy parameter for the Laplace mechanism;
                    also determines q = 1 / (e^epsilon + 1)
        delta:      total TCT failure probability (delta_infinity = delta);
                    this is the system-level parameter from Theorem 3, not
                    the user's initial budget
    """

    def __init__(self, epsilon: float, delta: float):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        self.epsilon = epsilon
        self.delta = delta
        # Lemma 2.3 (NotPrior target): q = 1 / (e^epsilon + 1)
        self.q = 1.0 / (math.exp(epsilon) + 1.0)

        # Per-user state: cx maps user_id -> prepaid hit count (Fx.cx in paper).
        # cx = 0: not yet registered.
        # cx = k (k >= 1): K_k compositions have been prepaid.
        self._cx: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Core mathematical primitives (public for H2 analysis)
    # ------------------------------------------------------------------

    def alpha(self, i: int) -> float:
        """
        Dynamic alpha for step i (Theorem 3).

        delta_i = 6 / (pi^2 * i^2) * delta        [Zeta-2 allocation]
        beta_i  = ln(1 / delta_i)
        alpha_i = (beta_i + sqrt(beta_i^2 + 2 * beta_i * i)) / i
                  [positive root of: i * alpha^2 - 2*beta_i*alpha - 2*beta_i = 0]

        Properties: alpha_i > 0, alpha_i is decreasing in i (Remark 4).
        """
        if i < 1:
            raise ValueError(f"i must be >= 1, got {i}")
        delta_i = (6.0 / (math.pi ** 2 * i ** 2)) * self.delta
        beta_i = math.log(1.0 / delta_i)
        discriminant = beta_i ** 2 + 2.0 * beta_i * i
        return (beta_i + math.sqrt(discriminant)) / i

    def compositions(self, k: int) -> float:
        """
        K_k = (1 + alpha_k) / q * k

        The prepaid number of epsilon-DP compositions required to cover k
        target hits, with failure probability delta (Theorem 3 / Algorithm 8).
        K_0 = 0 by convention.
        """
        if k == 0:
            return 0.0
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        alpha_k = self.alpha(k)
        return (1.0 + alpha_k) / self.q * k

    # ------------------------------------------------------------------
    # IDP filter operations (Algorithms 4/7 and 5/8)
    # ------------------------------------------------------------------

    def _register(self, user: User) -> tuple[float, float]:
        """
        RegisterForTCT (Algorithms 4/7): one-time registration charge.

        Charges K_1 * epsilon from the user's epsilon budget and the full
        system delta from the user's delta budget (pre-paying delta_infinity).
        Sets cx = 1.
        """
        k1 = self.compositions(1)
        self._cx[user.user_id] = 1
        return (k1 * self.epsilon, self.delta)

    def _charge_hit(self, user: User) -> tuple[float, float]:
        """
        ChargeTargetHit (Algorithm 8): incremental charge on each hit.

        Extends prepaid budget from K_cx to K_{cx+1} compositions.
        Delta cost is 0 (delta is fully pre-paid at registration).

        Implements Algorithm 8 (not Algorithm 5, which has a sign error):
          cost = (K_{cx+1} - K_cx) * epsilon
        This telescopes correctly to K_{k+1} * epsilon after k hits.
        """
        cx = self._cx[user.user_id]
        k_next = self.compositions(cx + 1)
        k_curr = self.compositions(cx)
        self._cx[user.user_id] = cx + 1
        return ((k_next - k_curr) * self.epsilon, 0.0)

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
        IDP filter charge for one round (Algorithm 6, lines 4–12).

        Order within a round:
          1. RegisterForTCT if cx == 0 (pre-pays K_1*eps + delta).
          2. ChargeTargetHit if is_top AND user contributed to the hit.

        Both can happen in the same round (user's first round is also a hit).
        Non-top rounds cost zero after registration — this is the key
        efficiency advantage of TCT over standard IDP composition.
        """
        eps_cost = 0.0
        delta_cost = 0.0

        if self._cx.get(user.user_id, 0) == 0:
            r_eps, r_delta = self._register(user)
            eps_cost += r_eps
            delta_cost += r_delta

        if is_top and contributed:
            h_eps, h_delta = self._charge_hit(user)
            eps_cost += h_eps
            delta_cost += h_delta

        return (eps_cost, delta_cost)

    # ------------------------------------------------------------------
    # Analysis helpers (used in H2 experiments)
    # ------------------------------------------------------------------

    def hit_count(self, user_id: int) -> int:
        """Current cx for user: the prepaid hit count (Fx.cx in the paper)."""
        return self._cx.get(user_id, 0)

    def epsilon_prepaid(self, user_id: int) -> float:
        """Total epsilon charged to this user so far: K_cx * epsilon."""
        return self.compositions(self.hit_count(user_id)) * self.epsilon

    def theoretical_bound(self, k_hits: int) -> float:
        """
        K_k: the theoretical upper bound on the number of private accesses
        needed to realize k actual target hits (Theorem 3). Used for H2
        Theoretical vs Empirical plot.
        """
        return self.compositions(k_hits)
