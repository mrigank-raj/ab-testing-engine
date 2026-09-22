"""
Decision Engine
=================
Combines every module built so far (SRM, sequential testing, CUPED, CATE)
into ONE function that a non-technical stakeholder can actually use:
given an experiment's data, return a single verdict - SHIP, KILL, HOLD,
or NEEDS_MORE_DATA - with the reasoning spelled out.

Why this is the step that makes it "a tool" and not "a stats library":
Nobody wants to run six scripts and interpret six separate outputs
themselves. A PM wants one answer they can act on. This is that answer.

Decision priority order (and why this order, specifically):
1. SRM check FIRST, always. If the traffic split is broken, every
   downstream result is potentially biased - there is no point running
   any other test until the instrumentation bug is fixed. This is a
   hard gate, not a soft warning.
2. Sequential test status - tells us whether it's statistically SAFE to
   stop yet, given that the team may be checking this dashboard daily.
3. CUPED-adjusted effect estimate is preferred over the raw one whenever
   available, since it's strictly lower-variance for the same data.
4. CATE/uplift context is attached as an ADVISORY note, not a gate - it
   informs a ship decision (e.g. "ship, but consider targeting only the
   responsive segment") rather than blocking one.
"""

from dataclasses import dataclass, field
import pandas as pd

from srm import check_srm
from frequentist import two_proportion_ztest
from sequential import run_sequential_test
from cuped import compare_raw_vs_cuped


@dataclass
class Decision:
    verdict: str  # "SHIP" | "KILL" | "HOLD" | "NEEDS_MORE_DATA" | "INVALID_SRM"
    reasoning: list[str] = field(default_factory=list)
    effect_estimate: float | None = None
    confidence_interval: tuple | None = None
    p_value: float | None = None
    srm_check_passed: bool = True
    sequential_safe_to_stop: bool = False


def evaluate_experiment(
    df: pd.DataFrame,
    alpha: float = 0.05,
    min_practical_effect: float = 0.0,
    uplift_context: str | None = None,
    planned_duration_days: int | None = None,
) -> Decision:
    """
    The single entry point. Expects df with columns matching the
    simulator's schema: group, converted, pre_experiment_activity,
    day_assigned (day_assigned only needed if you want the sequential
    check; if absent, that gate is skipped with a note).

    min_practical_effect: the smallest lift that's actually WORTH shipping
    for, even if statistically significant (a statistically significant
    but practically tiny effect might not be worth the engineering cost
    of shipping - this lets the decision engine account for that).

    uplift_context: optional plain-text note from the CATE model (Step 7),
    e.g. "44% of users show positive predicted uplift" - attached as
    advisory context, not used to gate the decision.

    planned_duration_days: the experiment's PLANNED total duration (e.g.
    from the power analysis in Step 2). Without this, a genuinely null
    effect would report NEEDS_MORE_DATA forever, since mSPRT correctly
    never crosses the threshold when there's truly nothing there. Passing
    this lets the engine correctly conclude "no effect found, HOLD" once
    the planned duration is reached without significance - a business
    cannot wait on an inconclusive test indefinitely.
    """
    reasoning = []

    # --- Gate 1: SRM check (hard gate) ---
    srm_result = check_srm(df)
    if srm_result.srm_detected:
        reasoning.append(
            f"SRM DETECTED: observed split {srm_result.observed_split:.2%} vs expected 50% "
            f"(p={srm_result.p_value}). Every result below is potentially biased until this "
            f"is fixed - check platform/segment-level diagnostics to localize the bug."
        )
        return Decision(
            verdict="INVALID_SRM",
            reasoning=reasoning,
            srm_check_passed=False,
        )
    reasoning.append("SRM check passed - traffic split matches expected ratio.")

    # --- Gate 2: Sequential test status (is it safe to stop / conclude yet?) ---
    sequential_safe = False
    reached_planned_duration = False
    if "day_assigned" in df.columns:
        seq_results = run_sequential_test(df, alpha=alpha)
        final_day_result = seq_results[-1]
        sequential_safe = final_day_result.reject_null
        reached_planned_duration = (
            planned_duration_days is not None and final_day_result.day >= planned_duration_days - 1
        )
        if sequential_safe:
            reasoning.append(
                f"Sequential test (mSPRT) has crossed the significance threshold as of day "
                f"{final_day_result.day} - safe to conclude even if this was checked daily."
            )
        elif reached_planned_duration:
            reasoning.append(
                f"Sequential test has NOT crossed the threshold, but the experiment has reached "
                f"its planned duration ({planned_duration_days} days) - concluding as inconclusive "
                f"rather than waiting indefinitely."
            )
        else:
            reasoning.append(
                f"Sequential test (mSPRT) has NOT yet crossed the significance threshold "
                f"(day {final_day_result.day}). Continuing to collect data is still safe; "
                f"stopping now and forcing a verdict would risk a false conclusion."
            )
    else:
        reasoning.append("No day-level data provided - sequential check skipped.")

    # --- Effect estimate: prefer CUPED-adjusted over raw ---
    cuped_result = compare_raw_vs_cuped(df, alpha=alpha)
    effect = cuped_result.cuped_absolute_lift
    p_value = cuped_result.p_value_cuped
    reasoning.append(
        f"CUPED-adjusted effect estimate: {effect:.2%} (p={p_value}), "
        f"{cuped_result.variance_reduction_pct:.1f}% lower variance than the raw estimate."
    )

    # --- Attach CATE/uplift context as advisory, not a gate ---
    if uplift_context:
        reasoning.append(f"Uplift model context: {uplift_context}")

    # --- Final verdict logic ---
    if not sequential_safe and not reached_planned_duration and "day_assigned" in df.columns:
        verdict = "NEEDS_MORE_DATA"
    elif p_value >= alpha:
        verdict = "HOLD"
        reasoning.append("Effect is not statistically significant - insufficient evidence to ship or kill.")
    elif effect < 0:
        verdict = "KILL"
        reasoning.append(f"Effect is significant and NEGATIVE ({effect:.2%}) - recommend killing this feature.")
    elif abs(effect) < min_practical_effect:
        verdict = "HOLD"
        reasoning.append(
            f"Effect is statistically significant but below the minimum practical threshold "
            f"({min_practical_effect:.2%}) - likely not worth the engineering cost to ship."
        )
    else:
        verdict = "SHIP"
        reasoning.append(f"Effect is significant and positive ({effect:.2%}) - recommend shipping.")

    return Decision(
        verdict=verdict,
        reasoning=reasoning,
        effect_estimate=effect,
        p_value=p_value,
        srm_check_passed=True,
        sequential_safe_to_stop=sequential_safe,
    )


def print_decision(decision: Decision):
    print(f"\n{'=' * 55}")
    print(f"VERDICT: {decision.verdict}")
    print(f"{'=' * 55}")
    for i, line in enumerate(decision.reasoning, 1):
        print(f"{i}. {line}")
    if decision.effect_estimate is not None:
        print(f"\nEffect estimate: {decision.effect_estimate:.2%}  |  p-value: {decision.p_value}")


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    from simulator import simulate_experiment, SimulationConfig

    print("### SCENARIO 1: Clean experiment, clear positive effect ###")
    cfg1 = SimulationConfig(n_users=50_000, true_control_rate=0.10, true_lift=0.02, seed=42)
    df1 = simulate_experiment(cfg1)
    decision1 = evaluate_experiment(df1, planned_duration_days=14,
                                     uplift_context="44% of users show positive predicted uplift (Step 7 model)")
    print_decision(decision1)

    print("\n\n### SCENARIO 2: Broken traffic split (SRM bug on iOS) ###")
    cfg2 = SimulationConfig(n_users=50_000, biased_platform="iOS", segment_bias_strength=0.08, seed=1)
    df2 = simulate_experiment(cfg2)
    decision2 = evaluate_experiment(df2, planned_duration_days=14)
    print_decision(decision2)

    print("\n\n### SCENARIO 3: No real effect (null), experiment has run its full course ###")
    cfg3 = SimulationConfig(n_users=50_000, true_control_rate=0.10, true_lift=0.0, seed=99)
    df3 = simulate_experiment(cfg3)
    decision3 = evaluate_experiment(df3, planned_duration_days=14)
    print_decision(decision3)

    print("\n\n### SCENARIO 4: Early days, effect real but not enough data yet ###")
    cfg4 = SimulationConfig(n_users=3_000, true_control_rate=0.10, true_lift=0.02, n_days=14, seed=7)
    df4 = simulate_experiment(cfg4)
    # only look at the first 2 days worth of data to simulate an early check
    df4_early = df4[df4["day_assigned"] <= 1]
    decision4 = evaluate_experiment(df4_early, planned_duration_days=14)
    print_decision(decision4)
