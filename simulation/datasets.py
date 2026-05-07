"""
Synthetic heavy-tailed user population for the attribution simulation.

Uses Beta(alpha, beta) with alpha < 1 to model conversion rates. This gives a
J-shaped distribution: most users have rates near zero (rarely convert), with a
heavy right tail representing the small fraction of highly active users. Because
rates live in [0, 1] by construction, no capping is needed and each user
contributes at most 1 conversion per round, keeping query sensitivity = 1.

"""

from dataclasses import dataclass

import numpy as np

from simulation.population import User, UserPopulation


@dataclass
class PopulationConfig:
    """
    Parameters for synthetic population generation.

    alpha, beta: shape parameters of Beta(alpha, beta).
        alpha < 1 gives heavy mass near 0 (sparse converters).
        beta > 1 further concentrates the distribution toward 0.
        Expected conversion rate = alpha / (alpha + beta).
    epsilon_budget: initial per-user epsilon privacy budget.
    delta_budget:   initial per-user delta privacy budget.
        For the Standard IDP baseline (pure Laplace), delta is never consumed,
        but it must be set for Adaptive TCT which pre-pays delta_infinity.
    """
    n_users: int
    alpha: float
    beta: float
    epsilon_budget: float
    delta_budget: float

    def __post_init__(self):
        if self.n_users <= 0:
            raise ValueError(f"n_users must be positive, got {self.n_users}")
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(
                f"alpha and beta must be positive, got alpha={self.alpha}, beta={self.beta}"
            )
        if self.epsilon_budget <= 0:
            raise ValueError(f"epsilon_budget must be positive, got {self.epsilon_budget}")
        if self.delta_budget < 0:
            raise ValueError(f"delta_budget must be non-negative, got {self.delta_budget}")

    @property
    def expected_rate(self) -> float:
        """Theoretical mean of Beta(alpha, beta) = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def rate_variance(self) -> float:
        """Theoretical variance of Beta(alpha, beta)."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def rate_std(self) -> float:
        return float(np.sqrt(self.rate_variance))


def generate_synthetic_population(
    config: PopulationConfig,
    rng: np.random.Generator,
) -> UserPopulation:
    """
    Sample N users with Beta-distributed conversion rates.

    Each user's conversion_rate is the probability that they generate an
    attributed conversion in a given round (Bernoulli model). All budgets
    are initialised identically from config.
    """
    rates = rng.beta(config.alpha, config.beta, size=config.n_users)

    users = [
        User(
            user_id=i,
            epsilon_budget=config.epsilon_budget,
            delta_budget=config.delta_budget,
            conversion_rate=float(rates[i]),
        )
        for i in range(config.n_users)
    ]
    return UserPopulation(users)


def describe_population(population: UserPopulation) -> dict:
    """
    Summary statistics for the conversion rate distribution.

    Use this to validate the dataset has the intended heavy-tailed shape
    before running main experiments (eval plan Step 1.3).
    """
    rates = np.array([u.conversion_rate for u in population.all_users])
    return {
        "n_users": int(len(rates)),
        "mean": float(np.mean(rates)),
        "std": float(np.std(rates)),
        "median": float(np.median(rates)),
        "p10": float(np.percentile(rates, 10)),
        "p25": float(np.percentile(rates, 25)),
        "p75": float(np.percentile(rates, 75)),
        "p90": float(np.percentile(rates, 90)),
        "p99": float(np.percentile(rates, 99)),
        "frac_below_0.05": float(np.mean(rates < 0.05)),
        "frac_below_0.20": float(np.mean(rates < 0.20)),
        "frac_above_0.50": float(np.mean(rates > 0.50)),
    }


def plot_conversion_rate_histogram(
    population: UserPopulation,
    config: PopulationConfig,
    ax=None,
    bins: int = 60,
):
    """
    Histogram of sampled conversion rates with the theoretical Beta density
    overlaid. Returns the matplotlib Axes for further customisation.

    Corresponds to eval plan Step 1.3: "Plot a histogram of user contribution
    counts" to validate the dataset shape.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import beta as beta_dist

    rates = np.array([u.conversion_rate for u in population.all_users])

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.hist(rates, bins=bins, density=True, alpha=0.55, color="steelblue",
            label="Sampled rates")

    x = np.linspace(1e-4, 1 - 1e-4, 500)
    ax.plot(x, beta_dist.pdf(x, config.alpha, config.beta),
            color="crimson", lw=2,
            label=rf"Beta($\alpha$={config.alpha}, $\beta$={config.beta})")

    ax.axvline(config.expected_rate, color="darkorange", linestyle="--", lw=1.5,
               label=f"E[rate] = {config.expected_rate:.3f}")

    ax.set_xlabel("Conversion rate")
    ax.set_ylabel("Density")
    ax.set_title(
        f"User conversion rate distribution  "
        f"(N={config.n_users}, "
        rf"$\alpha$={config.alpha}, $\beta$={config.beta})"
    )
    ax.legend()
    return ax
