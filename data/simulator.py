"""
Experiment Data Simulator
==========================
Generates synthetic A/B test data with a KNOWN ground-truth effect size.

Why this exists (don't skip this comment when explaining the project):
Real-world A/B test datasets never tell you the true effect — you only ever
see one realization of noisy data. That means you can never verify whether
a testing METHOD (t-test, sequential test, CUPED-adjusted test, etc.) is
actually correct, only that it produced *a* result.

This simulator solves that by letting us set the true effect ourselves,
generate data, run our methods on it, and check whether they recover the
effect we planted — and, just as importantly, whether they correctly stay
quiet (no false positive) when we set the true effect to exactly zero.

That's what makes the "methodology validation" notebook possible later.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    n_users: int = 100_000          # total users across both arms
    true_control_rate: float = 0.10  # baseline conversion rate
    true_lift: float = 0.02          # ABSOLUTE lift in conversion rate for treatment
                                      # e.g. 0.02 means treatment converts at 12% vs 10% control
    traffic_split: float = 0.5       # fraction assigned to treatment
    srm_bias: float = 0.0            # 0 = no SRM. e.g. 0.05 skews split by 5 percentage points
    covariate_correlation: float = 0.6  # correlation between pre-experiment activity and outcome
                                         # (needed later for CUPED to have something to work with)
    n_days: int = 14                 # experiment duration, needed for sequential testing simulation
    seed: int | None = None


def simulate_experiment(config: SimulationConfig) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per user:
        user_id, day_assigned, group ('control'/'treatment'),
        pre_experiment_activity (covariate for CUPED),
        converted (0/1 outcome)

    The true lift is baked in at generation time — we control it, so later
    we can check whether our statistical methods correctly recover it.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_users

    # --- Assignment, with optional injected SRM ---
    # SRM = the actual split ratio deviates from the intended split due to a
    # bug (bot filtering, logging drop, redirect failure, etc.)
    effective_split = config.traffic_split + config.srm_bias
    effective_split = np.clip(effective_split, 0.01, 0.99)
    group = rng.choice(
        ["control", "treatment"],
        size=n,
        p=[1 - effective_split, effective_split],
    )

    # --- Pre-experiment covariate (e.g. prior 30-day activity/spend) ---
    # Deliberately correlated with the outcome so CUPED has a real signal
    # to use for variance reduction later. Independent of treatment
    # assignment (as it must be — it's PRE-experiment).
    pre_activity = rng.normal(loc=50, scale=15, size=n)
    pre_activity = np.clip(pre_activity, 0, None)

    # --- Outcome generation ---
    # Base propensity to convert is a function of the covariate (so the
    # covariate is genuinely predictive) plus the treatment effect.
    base_logit = -2.0 + config.covariate_correlation * (pre_activity - 50) / 15
    treatment_bump = np.where(
        group == "treatment",
        np.log(
            (config.true_control_rate + config.true_lift) / (1 - config.true_control_rate - config.true_lift)
        ) - np.log(config.true_control_rate / (1 - config.true_control_rate)),
        0.0,
    )
    logit = base_logit + treatment_bump
    prob_convert = 1 / (1 + np.exp(-logit))
    # Rescale so control group's average matches true_control_rate exactly
    control_mask = group == "control"
    scale_factor = config.true_control_rate / prob_convert[control_mask].mean()
    prob_convert = np.clip(prob_convert * scale_factor, 0, 1)

    converted = rng.binomial(1, prob_convert)

    # --- Spread assignment across days (needed for sequential testing) ---
    day_assigned = rng.integers(0, config.n_days, size=n)

    df = pd.DataFrame({
        "user_id": np.arange(n),
        "day_assigned": day_assigned,
        "group": group,
        "pre_experiment_activity": pre_activity,
        "converted": converted,
    })
    return df


def simulate_many_null_experiments(n_experiments: int, base_config: SimulationConfig) -> list[pd.DataFrame]:
    """
    Generate many experiments where true_lift = 0 (the null is TRUE).
    This is exactly what Step 5's false-positive-rate validation needs:
    if a method is well-calibrated, it should falsely reject the null in
    ~alpha% of these (e.g. ~5% at alpha=0.05) — not more.
    """
    experiments = []
    for i in range(n_experiments):
        cfg = SimulationConfig(**{**base_config.__dict__, "true_lift": 0.0, "seed": i})
        experiments.append(simulate_experiment(cfg))
    return experiments


if __name__ == "__main__":
    # Quick sanity check when run directly
    cfg = SimulationConfig(n_users=50_000, true_control_rate=0.10, true_lift=0.02, seed=42)
    df = simulate_experiment(cfg)
    print(df.head())
    print("\nGroup sizes:\n", df["group"].value_counts())
    print("\nObserved conversion rate by group:\n", df.groupby("group")["converted"].mean())
    print(f"\nTrue effect planted: {cfg.true_lift:.3f} "
          f"(control={cfg.true_control_rate:.3f}, treatment={cfg.true_control_rate + cfg.true_lift:.3f})")
