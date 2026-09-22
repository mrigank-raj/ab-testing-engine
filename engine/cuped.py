"""
CUPED - Controlled-experiment Using Pre-Experiment Data
===========================================================
Reduces variance in the outcome metric using a pre-experiment covariate,
shrinking confidence intervals WITHOUT collecting more data. Originates
from a well-known Microsoft experimentation platform paper (Deng et al.).

Why this matters in practice:
Teams are traffic-constrained. Every experiment competes with every other
experiment for the same pool of users. If you can detect the same effect
with fewer users (or in less time), that's a direct, quantifiable cost
saving - which is exactly the kind of argument that gets an experimentation
platform adopted by a real team.

The core idea:
Some of the variance in a user's outcome (did they convert?) is just
"this user was always more likely to convert" - a fact you already knew
BEFORE the experiment even started, from their prior behavior. CUPED
statistically removes that predictable part, so what's left is closer to
pure signal from the treatment itself.

CUPED-adjusted outcome:
    Y_cuped = Y - theta * (X - mean(X))
where X is the pre-experiment covariate and theta is chosen to minimize
the variance of Y_cuped (theta = Cov(Y, X) / Var(X), the OLS regression
coefficient of Y on X).

Critically: theta is estimated ONCE across both groups combined (never
per-group), and X must be measured BEFORE treatment assignment - otherwise
you'd be adjusting for something the treatment itself could have affected,
which would bias the result rather than just reducing its variance.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats


@dataclass
class CUPEDResult:
    theta: float
    control_n: int
    treatment_n: int
    raw_absolute_lift: float
    raw_ci_width: float
    cuped_absolute_lift: float
    cuped_ci_width: float
    variance_reduction_pct: float
    p_value_raw: float
    p_value_cuped: float


def apply_cuped(df: pd.DataFrame, outcome_col: str = "converted",
                 covariate_col: str = "pre_experiment_activity") -> pd.DataFrame:
    """
    Adds a 'cuped_adjusted' column to the dataframe.
    theta is estimated across the WHOLE dataset (both groups combined) -
    this is important: using a shared theta is what keeps the adjustment
    unbiased, since we're not letting the treatment group's own outcome
    data influence its own adjustment.
    """
    df = df.copy()
    Y = df[outcome_col].values
    X = df[covariate_col].values

    covariance = np.cov(Y, X)[0, 1]
    variance_x = np.var(X)
    theta = covariance / variance_x if variance_x > 0 else 0.0

    df["cuped_adjusted"] = Y - theta * (X - X.mean())
    return df, theta


def compare_raw_vs_cuped(df: pd.DataFrame, alpha: float = 0.05) -> CUPEDResult:
    """
    Runs the same two-sample test on both the raw outcome and the
    CUPED-adjusted outcome, and reports the confidence interval width
    reduction - this IS the demonstrable value of CUPED.
    """
    df_adj, theta = apply_cuped(df)

    def _test(col):
        control = df_adj[df_adj["group"] == "control"][col]
        treatment = df_adj[df_adj["group"] == "treatment"][col]
        n1, n2 = len(control), len(treatment)
        p1, p2 = control.mean(), treatment.mean()
        se = np.sqrt(control.var(ddof=1) / n1 + treatment.var(ddof=1) / n2)
        diff = p2 - p1
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_width = 2 * z_crit * se
        z = diff / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        return diff, ci_width, p_value, n1, n2

    raw_lift, raw_width, raw_p, n1, n2 = _test("converted")
    cuped_lift, cuped_width, cuped_p, _, _ = _test("cuped_adjusted")

    variance_reduction = 1 - (cuped_width / raw_width) ** 2  # CI width scales with sqrt(variance)

    return CUPEDResult(
        theta=round(theta, 5),
        control_n=n1,
        treatment_n=n2,
        raw_absolute_lift=round(raw_lift, 5),
        raw_ci_width=round(raw_width, 5),
        cuped_absolute_lift=round(cuped_lift, 5),
        cuped_ci_width=round(cuped_width, 5),
        variance_reduction_pct=round(variance_reduction * 100, 2),
        p_value_raw=round(raw_p, 6),
        p_value_cuped=round(cuped_p, 6),
    )


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    from simulator import simulate_experiment, SimulationConfig

    print("=== CUPED vs Raw comparison ===")
    print("Using the same pre-experiment covariate built into the simulator (Step 1)\n")

    cfg = SimulationConfig(
        n_users=20_000,
        true_control_rate=0.10,
        true_lift=0.01,          # deliberately a SMALLER effect than earlier tests -
                                  # this is where CUPED's value shows up most clearly
        covariate_correlation=3.0,  # tuned so the covariate has a realistic, meaningful
                                     # correlation with a BINARY outcome (see note below)
        seed=7,
    )
    df = simulate_experiment(cfg)
    result = compare_raw_vs_cuped(df)

    print(f"Theta (regression coefficient): {result.theta}")
    print(f"\n{'':20s}{'RAW':>15s}{'CUPED-ADJUSTED':>18s}")
    print(f"{'Absolute lift':20s}{result.raw_absolute_lift:>15.4%}{result.cuped_absolute_lift:>18.4%}")
    print(f"{'95% CI width':20s}{result.raw_ci_width:>15.4%}{result.cuped_ci_width:>18.4%}")
    print(f"{'p-value':20s}{result.p_value_raw:>15.5f}{result.p_value_cuped:>18.5f}")
    print(f"\nVariance reduction from CUPED: {result.variance_reduction_pct:.1f}%")
    print(f"(This means CUPED could detect the same effect with meaningfully fewer users,")
    print(f" or detect a smaller effect with the SAME number of users.)")

    # Theoretical sanity check: CUPED's variance reduction should equal
    # approximately rho^2 (correlation squared) between the covariate and
    # outcome. This is a known closed-form result - checking it confirms
    # the implementation is mathematically correct, not just plausible-looking.
    actual_corr = df["pre_experiment_activity"].corr(df["converted"])
    theoretical_reduction = (actual_corr ** 2) * 100
    print(f"\n--- Sanity check ---")
    print(f"Actual covariate-outcome correlation: {actual_corr:.4f}")
    print(f"Theoretical variance reduction (rho^2): {theoretical_reduction:.2f}%")
    print(f"Observed variance reduction:            {result.variance_reduction_pct:.2f}%")
    print(f"(These should be close - confirms the CUPED math is implemented correctly)")

    print(f"\nNOTE: CUPED's benefit is smaller on binary outcomes (like conversion) than on")
    print(f"continuous metrics (like revenue), because a single pre-experiment variable")
    print(f"rarely correlates strongly with a yes/no outcome. This is a known, honest")
    print(f"limitation worth stating explicitly rather than overselling the number.")
