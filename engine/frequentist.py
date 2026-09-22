"""
Baseline Frequentist Testing
==============================
The standard statistical tests every stakeholder already understands:
two-proportion z-test (equivalent to a t-test for binary conversion
outcomes), chi-square test of independence, and confidence intervals.

Why this module isn't "novel" and that's fine:
This is the credibility floor of the whole project. Every experimentation
report a PM reads speaks this language. The interesting parts of this
project (sequential testing, CUPED, CATE) all get COMPARED against this
baseline later — so it has to be correct and well-tested first.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats


@dataclass
class FrequentistResult:
    control_n: int
    treatment_n: int
    control_rate: float
    treatment_rate: float
    absolute_lift: float
    relative_lift: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool
    alpha: float


def two_proportion_ztest(df: pd.DataFrame, alpha: float = 0.05) -> FrequentistResult:
    """
    Standard test for comparing conversion rates between two groups.
    Expects df with columns: 'group' ('control'/'treatment'), 'converted' (0/1)

    This is mathematically equivalent to a chi-square test with 1 degree
    of freedom for a 2x2 table - included separately below for stakeholders
    who expect to see "chi-square" by name.
    """
    control = df[df["group"] == "control"]["converted"]
    treatment = df[df["group"] == "treatment"]["converted"]

    n1, n2 = len(control), len(treatment)
    p1, p2 = control.mean(), treatment.mean()

    # Pooled proportion under the null hypothesis (no difference)
    p_pool = (control.sum() + treatment.sum()) / (n1 + n2)
    se_null = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    z = (p2 - p1) / se_null
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))  # two-sided

    # Confidence interval uses the UNPOOLED standard error (standard practice
    # for CI construction, distinct from the pooled SE used for the p-value)
    se_diff = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p2 - p1
    ci_lower = diff - z_crit * se_diff
    ci_upper = diff + z_crit * se_diff

    return FrequentistResult(
        control_n=n1,
        treatment_n=n2,
        control_rate=round(p1, 4),
        treatment_rate=round(p2, 4),
        absolute_lift=round(diff, 4),
        relative_lift=round(diff / p1, 4) if p1 > 0 else float("nan"),
        p_value=round(p_value, 5),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        significant=p_value < alpha,
        alpha=alpha,
    )


def chi_square_test(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Chi-square test of independence on the 2x2 contingency table
    (group x converted). Given by name because many stakeholders expect
    to see it explicitly, even though it's mathematically equivalent to
    the two-proportion z-test above for a 2x2 table.
    """
    contingency = pd.crosstab(df["group"], df["converted"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    return {
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 5),
        "degrees_of_freedom": dof,
        "significant": p_value < alpha,
        "contingency_table": contingency.to_dict(),
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    from simulator import simulate_experiment, SimulationConfig

    print("=== TEST 1: Data WITH a real effect (true_lift=0.02) ===")
    cfg = SimulationConfig(n_users=50_000, true_control_rate=0.10, true_lift=0.02, seed=42)
    df = simulate_experiment(cfg)

    result = two_proportion_ztest(df)
    print(f"Control rate:   {result.control_rate:.2%}  (n={result.control_n:,})")
    print(f"Treatment rate: {result.treatment_rate:.2%}  (n={result.treatment_n:,})")
    print(f"Absolute lift:  {result.absolute_lift:.2%}  (true planted effect: 2.00%)")
    print(f"95% CI:         [{result.ci_lower:.2%}, {result.ci_upper:.2%}]")
    print(f"p-value:        {result.p_value}")
    print(f"Significant at alpha=0.05? {result.significant}  <- should be True")

    chi2_result = chi_square_test(df)
    print(f"\nChi-square p-value: {chi2_result['p_value']} (should match z-test p-value closely)")

    print("\n=== TEST 2: Data with NO real effect (true_lift=0.0) - checking for false positive ===")
    cfg_null = SimulationConfig(n_users=50_000, true_control_rate=0.10, true_lift=0.0, seed=99)
    df_null = simulate_experiment(cfg_null)
    result_null = two_proportion_ztest(df_null)
    print(f"Absolute lift:  {result_null.absolute_lift:.2%}  (true planted effect: 0.00%)")
    print(f"p-value:        {result_null.p_value}")
    print(f"Significant at alpha=0.05? {result_null.significant}  <- should usually be False")
