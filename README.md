# A/B Testing Decision Engine

An experimentation decision engine that goes beyond "is the p-value below 0.05?" — it tells product teams whether they can **trust** a result, whether it's **safe to stop testing yet**, and **which users** it actually worked for. Built to address the three most common ways real A/B tests mislead teams: broken traffic splits (SRM), peeking-induced false positives, and averaging away real segment-level differences.

**Live demo:** run the dashboard locally (see below) — a full working experimentation pipeline behind one interface.

---

## The problem this solves

Most A/B testing tutorials stop at "run a t-test, check if p < 0.05." In practice, that's where real experimentation mistakes begin:

- **Sample Ratio Mismatch (SRM)** — a broken redirect or logging bug skews the traffic split, silently invalidating the whole test
- **Peeking** — teams check dashboards daily and stop at the first significant-looking result, which inflates the true false-positive rate far above the nominal 5%
- **Averaging away heterogeneity** — a feature that helps power users and hurts new users can net out to "no effect," and a naive average kills a genuinely good feature

This project builds, and **proves with simulation and real data**, four techniques that address each of these directly.

---

## Architecture

```
ab-testing-engine/
├── data/
│   └── simulator.py          # synthetic experiment generator with KNOWN ground-truth effect
├── engine/
│   ├── power_analysis.py     # sample size / MDE calculator
│   ├── srm.py                 # Sample Ratio Mismatch detection + root-cause segmentation
│   ├── frequentist.py        # baseline t-test / chi-square / confidence intervals
│   ├── sequential.py         # mSPRT always-valid sequential testing
│   ├── cuped.py               # CUPED variance reduction
│   ├── cate.py                 # T-learner uplift/CATE modeling
│   └── decision_engine.py    # combines everything into one SHIP/KILL/HOLD verdict
├── api/
│   └── app.py                  # FastAPI backend serving the decision engine live
├── frontend/
│   └── dashboard.html          # interactive dashboard calling the live API
├── notebooks/
│   └── methodology_validation.ipynb   # proof: every claim below, validated with charts
└── README.md
```

The stats engine (`engine/`) is fully decoupled from the presentation layer (`api/`, `frontend/`) — the same decision logic could sit behind a CLI, a Slack bot, or a different UI without any changes to the core methods.

---

## What's novel here (vs. a typical portfolio A/B testing project)

| Most projects | This project |
|---|---|
| One dataset, one t-test | A simulator with **known ground-truth effects**, used to *prove* each method is statistically correct — not just plausible-looking |
| SRM check = yes/no flag | Automated **root-cause segmentation** — localizes *which platform/segment* the traffic imbalance concentrates in |
| Fixed-horizon testing only | **Sequential testing (mSPRT)** — simulated proof that naive daily peeking inflates false positives several-fold above nominal alpha, while mSPRT stays controlled |
| Raw variance | **CUPED** variance reduction, validated against its theoretical ρ² prediction |
| Average treatment effect only | **CATE/uplift modeling** on the real Criteo dataset (13.9M rows) — ranks users by treatment responsiveness, not just "did it work on average" |
| Notebook only | A **live API + dashboard** — an actual clickable tool, not just analysis |

---

## Key results (from `notebooks/methodology_validation.ipynb`)

- **Sequential testing:** naive daily-peeking testing showed a **22.7% false-positive rate** (vs. nominal 5%) under simulated repeated checking; mSPRT stayed at **1.0%** under the exact same conditions — a >4x inflation caught and controlled.
- **CUPED:** reduced confidence interval width by **11.6%**, matching the theoretical ρ² prediction almost exactly (11.61% observed vs. 11.61% predicted) — in one demo run, this was enough to tip a borderline result (p=0.0505) into significance (p=0.0490) without collecting a single additional user.
- **CATE/uplift modeling:** trained on the real Criteo Uplift dataset (13.9M rows); identified that only ~44% of targeted users showed positive predicted uplift — meaning ad spend on the remaining ~56% is plausibly wasted.
- **SRM root-cause detection:** correctly isolated an injected iOS-only traffic-split bug (58% vs. expected 50%) while correctly clearing unaffected platforms (Android, Web) in the same experiment.

---

## How to run it

### 1. Setup
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the decision engine directly
```bash
cd engine
python decision_engine.py
```

### 3. Run the live API + dashboard
```bash
cd api
uvicorn app:app --reload --port 8000
```
Then open `frontend/dashboard.html` in a browser, or visit `http://127.0.0.1:8000/docs` for the interactive API documentation.

### 4. Run the methodology validation notebook
```bash
jupyter notebook notebooks/methodology_validation.ipynb
```

---

## Tech stack

Python (NumPy, Pandas, SciPy, scikit-learn), FastAPI, HTML/CSS/JS (vanilla, no framework), Jupyter, the real-world [Criteo Uplift Modeling Dataset](https://huggingface.co/datasets/criteo/criteo-uplift).

---

## Honest limitations

- mSPRT's false-positive rate came in below the nominal 5% (conservative), not exactly at it — a known tradeoff of always-valid testing that could be tuned via the `tau` parameter for better power.
- CUPED's variance reduction is modest on binary outcomes (conversion) compared to continuous ones (revenue) — this is a known, general property of CUPED, not specific to this implementation.
- The Qini-like ranking metric used for CATE evaluation is a simplified proxy, not a full Qini curve — sufficient to confirm the model ranks users sensibly, not a publication-grade uplift metric.
