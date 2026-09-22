"""
Sample Ratio Mismatch (SRM) Detection + Root-Cause Diagnostics
==================================================================
SRM = the actual observed traffic split deviates from the intended split
(e.g. observed 53/47 when the experiment was configured as 50/50).

Why this matters:
SRM is one of the most common silent experiment killers in the real world.
It's usually caused by a BUG, not chance - a redirect failing for some
users, bot traffic hitting one arm harder, logging dropping events
asymmetrically on one platform. When SRM is present, your treatment effect
estimate is likely biased and CANNOT be trusted, no matter how significant
the p-value looks.

Most tools stop at "SRM detected: yes/no". The genuinely useful part -
and the novelty here - is automatically segmenting by available dimensions
(platform, device, geo, etc.) to localize WHERE the imbalance concentrates,
which is what actually helps an engineer find and fix the bug.
"""

import pandas as pd
from scipy import stats
from dataclasses import dataclass


# SRM checks use a much stricter alpha than normal significance testing.
# Rationale: under correct randomization, the split should almost NEVER
# deviate significantly by chance. A conventional 0.05 threshold would
# flag false alarms too often at large sample sizes. 0.001 is the
# threshold commonly used in industry (e.g. Microsoft's ExP platform).
SRM_ALPHA = 0.001


@dataclass
class SRMResult:
    expected_split: float
    observed_control_n: int
    observed_treatment_n: int
    observed_split: float
    chi2_statistic: float
    p_value: float
    srm_detected: bool


@dataclass
class SegmentDiagnosis:
    segment_name: str
    segment_value: str
    control_n: int
    treatment_n: int
    observed_split: float
    p_value: float
    flagged: bool


def check_srm(df: pd.DataFrame, expected_split: float = 0.5) -> SRMResult:
    """
    Chi-square goodness-of-fit test: does the observed group split match
    the intended split? Expects df with a 'group' column.
    """
    n_control = (df["group"] == "control").sum()
    n_treatment = (df["group"] == "treatment").sum()
    n_total = n_control + n_treatment

    observed = [n_control, n_treatment]
    expected = [n_total * (1 - expected_split), n_total * expected_split]

    chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    return SRMResult(
        expected_split=expected_split,
        observed_control_n=int(n_control),
        observed_treatment_n=int(n_treatment),
        observed_split=round(n_treatment / n_total, 4),
        chi2_statistic=round(chi2, 4),
        p_value=round(p_value, 6),
        srm_detected=p_value < SRM_ALPHA,
    )


def diagnose_srm_by_segment(
    df: pd.DataFrame,
    segment_columns: list[str],
    expected_split: float = 0.5,
) -> list[SegmentDiagnosis]:
    """
    For each segment column (e.g. 'platform'), break down the split ratio
    by each value within it (e.g. 'iOS', 'Android', 'Web') and run the SRM
    check independently on each. This is what localizes the bug instead of
    just confirming it exists.

    Returns results sorted by p-value ascending (most suspicious first).
    """
    diagnoses = []
    for col in segment_columns:
        for value in df[col].unique():
            subset = df[df[col] == value]
            result = check_srm(subset, expected_split=expected_split)
            diagnoses.append(SegmentDiagnosis(
                segment_name=col,
                segment_value=str(value),
                control_n=result.observed_control_n,
                treatment_n=result.observed_treatment_n,
                observed_split=result.observed_split,
                p_value=result.p_value,
                flagged=result.srm_detected,
            ))
    # Most suspicious (lowest p-value) first
    diagnoses.sort(key=lambda d: d.p_value)
    return diagnoses


def full_srm_report(df: pd.DataFrame, segment_columns: list[str], expected_split: float = 0.5) -> dict:
    """
    Combines the overall check with segment-level root-cause diagnosis
    into one report - this is what the decision engine (Step 8) will
    eventually consume.
    """
    overall = check_srm(df, expected_split)
    report = {"overall": overall, "segments": []}

    if overall.srm_detected:
        report["segments"] = diagnose_srm_by_segment(df, segment_columns, expected_split)

    return report


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    from simulator import simulate_experiment, SimulationConfig

    print("=== TEST 1: Clean experiment, no SRM ===")
    cfg_clean = SimulationConfig(n_users=50_000, seed=1)
    df_clean = simulate_experiment(cfg_clean)
    result_clean = check_srm(df_clean)
    print(f"Observed split: {result_clean.observed_split:.4f} (expected 0.5)")
    print(f"p-value: {result_clean.p_value}")
    print(f"SRM detected? {result_clean.srm_detected}  <- should be False")

    print("\n=== TEST 2: Bug injected ONLY on iOS (realistic scenario) ===")
    cfg_bug = SimulationConfig(
        n_users=50_000,
        biased_platform="iOS",
        segment_bias_strength=0.08,  # iOS redirect bug skews split by 8pp on that platform only
        seed=1,
    )
    df_bug = simulate_experiment(cfg_bug)

    overall = check_srm(df_bug)
    print(f"Overall observed split: {overall.observed_split:.4f} (expected 0.5)")
    print(f"Overall p-value: {overall.p_value}")
    print(f"Overall SRM detected? {overall.srm_detected}")

    print("\n--- Root-cause segmentation ---")
    diagnoses = diagnose_srm_by_segment(df_bug, segment_columns=["platform"])
    for d in diagnoses:
        flag = " <-- FLAGGED" if d.flagged else ""
        print(f"{d.segment_name}={d.segment_value:8s}  split={d.observed_split:.4f}  "
              f"p-value={d.p_value:.6f}{flag}")
