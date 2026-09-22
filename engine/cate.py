"""
CATE (Conditional Average Treatment Effect) / Uplift Modeling
==================================================================
The first genuinely ML-driven module in this project. Everything before
this was statistical INFERENCE on data with known structure. This is a
model that LEARNS from real data which users respond to treatment.

Why this matters (the actual business case):
The average treatment effect can hide the real story. A feature that
helps power users and annoys new users might net out to "no significant
effect" on average - and a team using only ATE would kill a genuinely
good feature. CATE answers a more useful question: not "did this work?"
but "who should get this?"

Approach: T-learner (two-model approach)
    1. Train a model on CONTROL users only -> predicts their natural
       (untreated) outcome probability
    2. Train a separate model on TREATMENT users only -> predicts their
       (treated) outcome probability
    3. Estimated individual treatment effect (uplift) for any user =
       treatment_model.predict(x) - control_model.predict(x)

This is the simplest defensible CATE method - more complex approaches
(X-learner, causal forests) exist but are harder to justify/explain in
an interview without deeper causal inference background. T-learner's
logic is easy to state precisely, which matters more than sophistication
for a portfolio project.

Dataset: Criteo Uplift Modeling Dataset (real randomized experiment data)
    - f0-f11: user features (dense, float)
    - treatment: 1 = shown ads, 0 = control
    - visit / conversion: binary outcomes
Fetched via the `scikit-uplift` library: sklift.datasets.fetch_criteo()
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


@dataclass
class UpliftModelResult:
    train_auc_control: float
    train_auc_treatment: float
    test_auc_control: float
    test_auc_treatment: float
    avg_predicted_uplift: float
    pct_users_positive_uplift: float
    qini_like_score: float  # simplified ranking-quality metric, see note below


class TLearnerUplift:
    """
    Two-model uplift estimator. Call .fit(X, treatment, y) then
    .predict_uplift(X) to get per-user estimated treatment effect.
    """

    def __init__(self, **model_kwargs):
        default_kwargs = dict(n_estimators=100, max_depth=3, random_state=42)
        default_kwargs.update(model_kwargs)
        self.model_control = GradientBoostingClassifier(**default_kwargs)
        self.model_treatment = GradientBoostingClassifier(**default_kwargs)

    def fit(self, X: pd.DataFrame, treatment: pd.Series, y: pd.Series):
        control_mask = treatment == 0
        treatment_mask = treatment == 1

        self.model_control.fit(X[control_mask], y[control_mask])
        self.model_treatment.fit(X[treatment_mask], y[treatment_mask])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Returns estimated individual treatment effect for each row: p(treated) - p(control)."""
        p_control = self.model_control.predict_proba(X)[:, 1]
        p_treatment = self.model_treatment.predict_proba(X)[:, 1]
        return p_treatment - p_control


def evaluate_uplift_model(
    X: pd.DataFrame,
    treatment: pd.Series,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[TLearnerUplift, UpliftModelResult, pd.DataFrame]:
    """
    Fits the T-learner, evaluates each sub-model's discrimination (AUC),
    and computes uplift for the test set. Returns the fitted model, a
    results summary, and a dataframe with per-user uplift scores attached
    (this last piece is what a real team would use to target users).
    """
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, y, test_size=test_size, random_state=random_state, stratify=treatment
    )

    model = TLearnerUplift()
    model.fit(X_train, t_train, y_train)

    # AUC on the two sub-models - a sanity check that each model is
    # actually learning something predictive, not just noise
    train_auc_c = roc_auc_score(y_train[t_train == 0], model.model_control.predict_proba(X_train[t_train == 0])[:, 1])
    train_auc_t = roc_auc_score(y_train[t_train == 1], model.model_treatment.predict_proba(X_train[t_train == 1])[:, 1])
    test_auc_c = roc_auc_score(y_test[t_test == 0], model.model_control.predict_proba(X_test[t_test == 0])[:, 1])
    test_auc_t = roc_auc_score(y_test[t_test == 1], model.model_treatment.predict_proba(X_test[t_test == 1])[:, 1])

    uplift_scores = model.predict_uplift(X_test)
    results_df = X_test.copy()
    results_df["treatment"] = t_test.values
    results_df["actual_outcome"] = y_test.values
    results_df["predicted_uplift"] = uplift_scores

    # Simplified qini-like score: correlation between predicted uplift rank
    # and actual outcome, split by treatment group. A properly implemented
    # Qini curve is more rigorous - this is a lightweight proxy sufficient
    # to confirm the model is ranking users sensibly, not a publication-grade metric.
    top_20_pct = results_df.nlargest(int(len(results_df) * 0.2), "predicted_uplift")
    bottom_20_pct = results_df.nsmallest(int(len(results_df) * 0.2), "predicted_uplift")
    top_actual_lift = (top_20_pct[top_20_pct.treatment == 1].actual_outcome.mean() -
                        top_20_pct[top_20_pct.treatment == 0].actual_outcome.mean())
    bottom_actual_lift = (bottom_20_pct[bottom_20_pct.treatment == 1].actual_outcome.mean() -
                           bottom_20_pct[bottom_20_pct.treatment == 0].actual_outcome.mean())
    qini_like = top_actual_lift - bottom_actual_lift  # should be POSITIVE if model ranks well

    result = UpliftModelResult(
        train_auc_control=round(train_auc_c, 4),
        train_auc_treatment=round(train_auc_t, 4),
        test_auc_control=round(test_auc_c, 4),
        test_auc_treatment=round(test_auc_t, 4),
        avg_predicted_uplift=round(float(uplift_scores.mean()), 5),
        pct_users_positive_uplift=round(float((uplift_scores > 0).mean()) * 100, 2),
        qini_like_score=round(float(qini_like), 5),
    )
    return model, result, results_df


if __name__ == "__main__":
    print("=== Attempting to load the REAL Criteo Uplift dataset (via Hugging Face) ===")
    try:
        # Hugging Face hosts Criteo's OFFICIAL copy of this dataset directly
        # (criteo/criteo-uplift) - more reliable than the older sklift S3
        # mirror, which has been intermittently unavailable.
        from datasets import load_dataset
        hf_data = load_dataset("criteo/criteo-uplift", split="train")
        df_full = hf_data.to_pandas()

        # Subsample for tractable local runtime - the full dataset is 13-25M
        # rows, far more than needed to demonstrate the modeling approach
        df_sample = df_full.sample(n=min(200_000, len(df_full)), random_state=42)

        feature_cols = [c for c in df_sample.columns if c.startswith("f")]
        X = df_sample[feature_cols].reset_index(drop=True)
        treatment = df_sample["treatment"].reset_index(drop=True)
        y = df_sample["visit"].reset_index(drop=True)
        print(f"Loaded real Criteo data: {X.shape[0]:,} rows (sampled from {len(df_full):,} total), {X.shape[1]} features")
    except Exception as e:
        print(f"Could not fetch real dataset ({type(e).__name__}: {e})")
        print("Falling back to a SYNTHETIC dataset with the same schema, purely to verify the")
        print("modeling code is correct. Replace this with the real load_dataset() call when")
        print("running with working internet access.\n")

        rng = np.random.default_rng(42)
        n = 50_000
        X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(12)})
        treatment = pd.Series(rng.choice([0, 1], size=n, p=[0.85, 0.15]), name="treatment")
        # Bake in genuine heterogeneity: f0 predicts BASE rate, f1 predicts UPLIFT specifically
        # (positive f1 -> treatment helps; negative f1 -> treatment does nothing / hurts)
        base_logit = -2.5 + 0.5 * X["f0"]
        uplift_logit = np.where(treatment == 1, 0.8 * X["f1"].clip(lower=0), 0.0)
        prob = 1 / (1 + np.exp(-(base_logit + uplift_logit)))
        y = pd.Series(rng.binomial(1, prob), name="visit")
        print(f"Synthetic fallback data: {X.shape[0]:,} rows, {X.shape[1]} features")

    print("\n=== Training T-learner uplift model ===")
    model, result, results_df = evaluate_uplift_model(X, treatment, y)

    print(f"\nControl sub-model AUC:   train={result.train_auc_control}, test={result.test_auc_control}")
    print(f"Treatment sub-model AUC: train={result.train_auc_treatment}, test={result.test_auc_treatment}")
    print(f"(AUC well above 0.5 confirms each model learned real signal, not noise)")

    print(f"\nAverage predicted uplift across test set: {result.avg_predicted_uplift:.4f}")
    print(f"% of users with POSITIVE predicted uplift: {result.pct_users_positive_uplift}%")
    print(f"Qini-like ranking score (top 20% vs bottom 20% actual lift): {result.qini_like_score:.4f}")
    print(f"(Positive score = model correctly ranks who benefits most, not just guessing)")