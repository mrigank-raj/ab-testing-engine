"""
Power Analysis / Sample Size Calculator
=========================================
Answers the question every experiment MUST answer before it launches:
"How many users do I need to reliably detect the effect I care about?"

Why this matters (real-world failure mode this prevents):
Teams that skip this step do one of two things wrong:
  1. Run a test with too few users -> a real effect exists but the test
     doesn't have enough statistical power to detect it, and the team
     wrongly concludes "no effect" and kills a good feature.
  2. Run a test far longer / with far more users than needed -> wastes
     traffic that could have been used for the next experiment.

This module uses the standard formula for a two-proportion z-test power
calculation (the most common case: comparing conversion rates between
two groups).
"""

import math
from dataclasses import dataclass
from scipy import stats


@dataclass
class PowerAnalysisResult:
    baseline_rate: float
    mde_absolute: float
    alpha: float
    power: float
    required_n_per_group: int
    required_n_total: int
    estimated_days: float | None = None


def required_sample_size(
    baseline_rate: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.80,
    daily_traffic: int | None = None,
) -> PowerAnalysisResult:
    """
    Calculate required sample size per group for a two-proportion test.

    Parameters
    ----------
    baseline_rate : control group's expected conversion rate (e.g. 0.10)
    mde_absolute : smallest absolute lift worth detecting (e.g. 0.02 for a
                   2 percentage point lift). Smaller MDE = needs more data.
    alpha : significance level (false positive rate we're willing to accept)
    power : probability of detecting the effect if it's really there
            (1 - false negative rate). Industry standard is 0.80.
    daily_traffic : optional, users/day available -> used to estimate
                     how many days the experiment will take to reach
                     the required sample size.

    Returns
    -------
    PowerAnalysisResult with required_n_per_group and, if daily_traffic
    given, estimated_days to complete.
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde_absolute

    # Z-scores for the two-sided test
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Pooled proportion for the standard two-proportion sample size formula
    p_pooled = (p1 + p2) / 2
    pooled_variance_term = 2 * p_pooled * (1 - p_pooled)

    numerator = (z_alpha * math.sqrt(pooled_variance_term) +
                 z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = mde_absolute ** 2

    n_per_group = math.ceil(numerator / denominator)
    n_total = n_per_group * 2

    estimated_days = None
    if daily_traffic:
        # assumes 50/50 split; adjust if your split differs
        estimated_days = round(n_total / daily_traffic, 1)

    return PowerAnalysisResult(
        baseline_rate=baseline_rate,
        mde_absolute=mde_absolute,
        alpha=alpha,
        power=power,
        required_n_per_group=n_per_group,
        required_n_total=n_total,
        estimated_days=estimated_days,
    )


def achieved_power(
    baseline_rate: float,
    mde_absolute: float,
    n_per_group: int,
    alpha: float = 0.05,
) -> float:
    """
    Reverse calculation: given a sample size you actually have (or plan to
    use), what power does that give you? Useful for checking "is my
    planned experiment even capable of detecting the effect I care about."
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde_absolute
    p_pooled = (p1 + p2) / 2

    se_null = math.sqrt(2 * p_pooled * (1 - p_pooled) / n_per_group)
    se_alt = math.sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z = (mde_absolute - z_alpha * se_null) / se_alt
    power = stats.norm.cdf(z)
    return round(power, 4)


if __name__ == "__main__":
    # Example: baseline conversion 10%, want to detect a 2pp lift (10% -> 12%)
    result = required_sample_size(
        baseline_rate=0.10,
        mde_absolute=0.02,
        alpha=0.05,
        power=0.80,
        daily_traffic=5000,
    )
    print("=== Power Analysis Result ===")
    print(f"Baseline rate:         {result.baseline_rate:.1%}")
    print(f"Minimum detectable effect: {result.mde_absolute:.1%} (absolute)")
    print(f"Significance level (alpha): {result.alpha}")
    print(f"Power:                 {result.power:.0%}")
    print(f"Required n per group:  {result.required_n_per_group:,}")
    print(f"Required n total:      {result.required_n_total:,}")
    print(f"Estimated days (at 5,000 users/day): {result.estimated_days}")

    # Sanity check: this should be close to 0.80 (the power we asked for)
    check = achieved_power(0.10, 0.02, result.required_n_per_group)
    print(f"\nSanity check - achieved power at that sample size: {check:.1%} "
          f"(should be close to 80%)")
