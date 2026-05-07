"""
Standard (non-adaptive) TCT accountant.

Alpha derivation (Cohen & Lyu 2023, Theorem B.4 / Remark B.5):
---------------------------------------------------------------
Theorem B.4 guarantees that Algorithm 1 run until τ target hits satisfies
(ε', δ) with ε' = (1+α)τε/q and δ* ≤ exp(-α²τ / 2(1+α)) for any α > 0.

To find the tightest (smallest K_τ = (1+α)τ/q) α satisfying δ* ≤ δ, set:

    exp(-α²τ / 2(1+α)) = δ
    α²τ / 2(1+α) = ln(1/δ) =: β
    τα² - 2βα - 2β = 0   [rearranged]

Positive root of this quadratic (Remark B.5 rearranged):

    α = (β + √(β² + 2βτ)) / τ,   β = ln(1/δ)

This is the minimum α (and therefore minimum K_τ) compatible with failure
probability δ.  Requires τ fixed upfront; cannot adaptively stop early.
"""

from __future__ import annotations

import math

from simulation.population import User
from .base import PrivacyAccountant


class StandardTCTAccountant(PrivacyAccountant):
    """
    Non-adaptive TCT accountant (Algorithms 4/5, Cohen & Lyu 2023).

    Args:
        epsilon:  per-query ε; also determines q = 1/(e^ε + 1)
        delta:    total failure probability (used in full, not split)
        tau:      predetermined number of top rounds the user will contribute to
    """

    def __init__(self, epsilon: float, delta: float, tau: int):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if tau < 1:
            raise ValueError(f"tau must be >= 1, got {tau}")

        self.epsilon = epsilon
        self.delta = delta
        self.tau = tau
        self.q = 1.0 / (math.exp(epsilon) + 1.0)

        # Cohen & Lyu 2023, Theorem B.4: δ*(τ,α) ≤ exp(-α²τ/2(1+α)).
        # Set δ*=δ, solve positive root of τα²−2βα−2β=0 (β=ln(1/δ)).
        beta = math.log(1.0 / delta)
        alpha_tau = (beta + math.sqrt(beta ** 2 + 2.0 * beta * tau)) / tau
        self.K_tau = (1.0 + alpha_tau) / self.q * tau  # (1+α)τ/q compositions
        self.alpha_tau = alpha_tau

        self._registered: set[int] = set()

    def charge(
        self,
        user: User,
        is_top: bool,
        contributed: bool,
        round_num: int,
    ) -> tuple[float, float]:
        """
        Charge K_tau * epsilon + delta at first appearance; nothing after.

        Non-adaptive TCT makes one upfront payment covering all tau tops.
        Unlike adaptive TCT there is no incremental ChargeTargetHit step.
        """
        if user.user_id not in self._registered:
            self._registered.add(user.user_id)
            return (self.K_tau * self.epsilon, self.delta)
        return (0.0, 0.0)

    def hit_count(self, user_id: int) -> int:
        return 1 if user_id in self._registered else 0

    def epsilon_prepaid(self, user_id: int) -> float:
        return self.K_tau * self.epsilon if user_id in self._registered else 0.0
