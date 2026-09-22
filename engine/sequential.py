"""
Sequential Testing / mSPRT (mixture Sequential Probability Ratio Test)
==========================================================================
Solves the single most common real-world experimentation mistake: PEEKING.

The problem this fixes:
A standard t-test / z-test is only valid if you decide your sample size
IN ADVANCE and look at the result exactly once, at the end. In practice,
almost nobody does this - teams check the dashboard every day and stop
the moment p < 0.05 shows up. This inflates the true false-positive rate
far above the nominal 5%, because you're effectively giving yourself many
independent chances to get a lucky false positive along the way.

mSPRT (and "always-valid inference" more broadly) fixes this by using a
test statistic that remains valid under continuous monitoring - you can
check it every single day, stop whenever you want, and the false-positive
guarantee still holds.

This module implements a Gaussian mixture SPRT for comparing two
conversion rates, and - critically - includes the SIMULATION PROOF that
demonstrates the fixed-horizon-with-peeking false positive rate inflation
and confirms mSPRT stays correctly calibrated. This proof is what makes
the project's core claim defensible, not just asserted.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SequentialTestResult:
    day: int
    n_control: int
    n_treatment: int
    observed_lift: float
    mixture_likelihood_ratio: float
    reject_null: bool  # True = statistically significant, safe to stop


def _daily_mixture_llr(df_so_far: pd.DataFrame, tau: float = 0.05) -> float:
    """
    Computes the mixture log-likelihood ratio for a two-proportion test
    at a single point in time, using a Gaussian mixture prior over the
    possible effect size (this is the "m" in mSPRT - the null is tested
    against a MIXTURE of alternatives, not one fixed alternative, which
    is what makes it valid at any stopping time).

    tau: prior standard deviation over the plausible effect size. Smaller
    tau = more sensitive to small effects but slower to accumulate
    evidence for large ones. 0.05 is a reasonable default for conversion
    rate experiments.
    """
    control = df_so_far[df_so_far["group"] == "control"]["converted"]
    treatment = df_so_far[df_so_far["group"] == "treatment"]["converted"]
    n1, n2 = len(control), len(treatment)
    if n1 < 30 or n2 < 30:
        return 0.0  # not enough data yet to compute anything meaningful

    p1, p2 = control.mean(), treatment.mean()
    diff = p2 - p1
    # Standard error of the difference (same as in the frequentist module)
    p_pool = (control.sum() + treatment.sum()) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0

    z = diff / se

    # Closed-form mixture likelihood ratio for a Gaussian mixture SPRT
    # (standard result: mixing a N(0, tau^2) prior over the true effect
    # against a Gaussian test statistic gives this closed form)
    variance_ratio = tau ** 2 / (se ** 2)
    llr = 0.5 * np.log(1 / (1 + variance_ratio)) + \
          0.5 * (z ** 2) * variance_ratio / (1 + variance_ratio)
    return llr


def run_sequential_test(
    df: pd.DataFrame,
    alpha: float = 0.05,
    tau: float = 0.05,
) -> list[SequentialTestResult]:
    """
    Runs the mSPRT test day by day, exactly as a team checking a dashboard
    daily would. Returns a result for EVERY day, with a stopping
    recommendation - this list IS what powers a "check anytime, safely"
    dashboard view.

    Expects df with columns: 'day_assigned', 'group', 'converted'
    """
    threshold = np.log(1 / alpha)  # standard SPRT decision boundary
    results = []
    max_day = df["day_assigned"].max()

    for day in range(max_day + 1):
        df_so_far = df[df["day_assigned"] <= day]
        llr = _daily_mixture_llr(df_so_far, tau=tau)

        control = df_so_far[df_so_far["group"] == "control"]["converted"]
        treatment = df_so_far[df_so_far["group"] == "treatment"]["converted"]
        lift = (treatment.mean() - control.mean()) if len(control) and len(treatment) else 0.0

        results.append(SequentialTestResult(
            day=day,
            n_control=len(control),
            n_treatment=len(treatment),
            observed_lift=round(float(lift), 5),
            mixture_likelihood_ratio=round(float(llr), 4),
            reject_null=bool(llr > threshold),
        ))
    return results


def naive_peeking_test(df: pd.DataFrame, alpha: float = 0.05) -> bool:
    """
    Simulates the COMMON MISTAKE: running a standard fixed-horizon
    z-test every day and stopping the first time p < alpha.
    This is what most teams actually do without realizing it's invalid.
    Returns True if this naive approach ever declared significance.
    """
    from scipy import stats
    max_day = df["day_assigned"].max()

    for day in range(max_day + 1):
        df_so_far = df[df["day_assigned"] <= day]
        control = df_so_far[df_so_far["group"] == "control"]["converted"]
        treatment = df_so_far[df_so_far["group"] == "treatment"]["converted"]
        n1, n2 = len(control), len(treatment)
        if n1 < 30 or n2 < 30:
            continue
        p1, p2 = control.mean(), treatment.mean()
        p_pool = (control.sum() + treatment.sum()) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        if se == 0:
            continue
        z = (p2 - p1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        if p_value < alpha:
            return True  # stopped early, declared "significant"
    return False


def mspt_ever_rejects(df: pd.DataFrame, alpha: float = 0.05, tau: float = 0.05) -> bool:
    """Returns True if mSPRT ever crosses the rejection threshold across all days checked."""
    results = run_sequential_test(df, alpha=alpha, tau=tau)
    return any(r.reject_null for r in results)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    from simulator import simulate_many_null_experiments, SimulationConfig

    N_EXPERIMENTS = 300  # null experiments (true effect = 0) to simulate

    print(f"=== Running {N_EXPERIMENTS} NULL experiments (true effect = 0) ===")
    print("A well-calibrated method should falsely declare 'significant' in ~5% of these.\n")

    base_config = SimulationConfig(n_users=20_000, true_control_rate=0.10, n_days=14)
    experiments = simulate_many_null_experiments(N_EXPERIMENTS, base_config)

    naive_false_positives = 0
    msprt_false_positives = 0

    for i, df in enumerate(experiments):
        if naive_peeking_test(df):
            naive_false_positives += 1
        if mspt_ever_rejects(df):
            msprt_false_positives += 1
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{N_EXPERIMENTS} experiments processed")

    naive_fpr = naive_false_positives / N_EXPERIMENTS
    msprt_fpr = msprt_false_positives / N_EXPERIMENTS

    print("\n=== RESULTS ===")
    print(f"Naive daily-peeking fixed-horizon test: {naive_false_positives}/{N_EXPERIMENTS} "
          f"false positives = {naive_fpr:.1%} false-positive rate")
    print(f"  (Nominal alpha was 5% - this shows how much peeking inflates it)")
    print(f"\nmSPRT (always-valid, safe to check daily): {msprt_false_positives}/{N_EXPERIMENTS} "
          f"false positives = {msprt_fpr:.1%} false-positive rate")
    print(f"  (Should stay close to the nominal 5%, unlike the naive approach above)")
