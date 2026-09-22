"""
FastAPI Backend
==================
Exposes the decision engine as a live, callable API - this is what turns
the project from "scripts I ran once" into "a service someone could
actually integrate into their pipeline."

Endpoints:
    GET  /health           - basic liveness check (standard for any real API)
    POST /simulate-and-analyze - generates a synthetic experiment with the
                                  given parameters, runs it through the full
                                  decision engine, returns the verdict as JSON

Run locally with:
    uvicorn app:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for FastAPI's auto-generated,
interactive API documentation (this alone is worth demoing live).
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "engine"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from data.simulator import simulate_experiment, SimulationConfig
from decision_engine import evaluate_experiment


app = FastAPI(
    title="A/B Testing Decision Engine",
    description=(
        "An experimentation decision engine that combines SRM detection, "
        "sequential (always-valid) testing, and CUPED variance reduction "
        "into a single ship/kill/hold recommendation."
    ),
    version="1.0.0",
)

# Allow the local frontend dashboard (opened as a file, or served from any
# origin during development) to call this API. In a real production
# deployment this would be locked down to a specific origin, not "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExperimentRequest(BaseModel):
    n_users: int = Field(50_000, description="Total users across both arms")
    true_control_rate: float = Field(0.10, description="Baseline conversion rate")
    true_lift: float = Field(0.02, description="Absolute lift in conversion rate for treatment")
    srm_bias: float = Field(0.0, description="Global traffic split bias to inject (0 = none)")
    biased_platform: Optional[str] = Field(None, description="Platform to inject segment-specific SRM into")
    segment_bias_strength: float = Field(0.0, description="Strength of the segment-specific SRM bias")
    n_days: int = Field(14, description="Experiment duration in days")
    alpha: float = Field(0.05, description="Significance level")
    min_practical_effect: float = Field(0.0, description="Smallest lift worth shipping for")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class DecisionResponse(BaseModel):
    verdict: str
    reasoning: list[str]
    effect_estimate: Optional[float]
    p_value: Optional[float]
    srm_check_passed: bool
    sequential_safe_to_stop: bool


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ab-testing-decision-engine"}


@app.post("/simulate-and-analyze", response_model=DecisionResponse)
def simulate_and_analyze(request: ExperimentRequest):
    """
    Generates a synthetic experiment with the given parameters and runs it
    through the full decision pipeline (SRM -> sequential -> CUPED -> verdict).

    This endpoint is what the frontend dashboard (Step 10) calls - a real
    production version of this API would instead accept ALREADY-COLLECTED
    experiment data rather than generating synthetic data on the fly; this
    version uses the simulator so the tool is self-contained and demoable
    without needing a live data pipeline behind it.
    """
    try:
        config = SimulationConfig(
            n_users=request.n_users,
            true_control_rate=request.true_control_rate,
            true_lift=request.true_lift,
            srm_bias=request.srm_bias,
            biased_platform=request.biased_platform,
            segment_bias_strength=request.segment_bias_strength,
            n_days=request.n_days,
            seed=request.seed,
        )
        df = simulate_experiment(config)

        decision = evaluate_experiment(
            df,
            alpha=request.alpha,
            min_practical_effect=request.min_practical_effect,
            planned_duration_days=request.n_days,
        )

        return DecisionResponse(
            verdict=decision.verdict,
            reasoning=decision.reasoning,
            effect_estimate=decision.effect_estimate,
            p_value=decision.p_value,
            srm_check_passed=decision.srm_check_passed,
            sequential_safe_to_stop=decision.sequential_safe_to_stop,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)