# LLM Fairness Eval

[![fairness-gate](https://github.com/sureshbujji/llm-fairness-eval/actions/workflows/fairness-gate.yml/badge.svg)](https://github.com/sureshbujji/llm-fairness-eval/actions/workflows/fairness-gate.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

Counterfactual fairness evaluation for LLM systems — the bias-testing companion to my [ai-qa-eval-harness](https://github.com/sureshbujji/ai-qa-eval-harness). Same task prompt, demographic signal swapped; systematic outcome gaps across groups are the bias signal.

## How it works

```
counterfactual prompts → model under test → fairness metrics → CI gate
```

1. **Counterfactual prompts** — `src/counterfactuals.py` expands 6 golden scenarios (hiring, performance reviews, credit advice, salary negotiation, medical triage) into one prompt variant per demographic group. Names are used as *signals* (widely perceived gender/ethnicity in US hiring-adjacent contexts), 8 groups × 3 names = 24 variants per case.
2. **Model backends** — `src/models.py`. Mock backend (deterministic, scripted — deliberately biased on hiring so the gate mechanics are exercised) or any OpenAI-compatible endpoint via `FAIRNESS_API_KEY` / `FAIRNESS_API_BASE` / `FAIRNESS_MODEL`.
3. **Metrics** — `src/metrics.py` (pure functions):
   - **Counterfactual consistency** — share of cases where every group variant produced the same outcome token.
   - **Demographic parity gap** — max minus min positive-outcome rate across groups.
   - **Sentiment gap** — max minus min mean lexicon sentiment across groups.
   - **Stereotype flag rate** — share of outputs containing stereotyped descriptors.
4. **CI gate** — `.github/workflows/fairness-gate.yml` runs unit tests plus the mock eval. The mock data is *deliberately* biased, so the gate is expected to **fail** — a passing gate on that data would mean the metrics went blind.

## Quickstart

```bash
git clone https://github.com/sureshbujji/llm-fairness-eval.git
cd llm-fairness-eval

# 1) Smoke test — no API key needed (mock backend, scripted bias)
python src/evaluator.py --mock
# -> variants=144 consistency=... parity_gap=0.333 ...

# 2) Real run — point at any OpenAI-compatible endpoint
cp .env.example .env   # fill in your key; never commit it
export $(cat .env | xargs)
python src/evaluator.py --fail-on-critical

# 3) Unit tests
python -m pytest tests/ -q
```

Reports land in `reports/` (`fairness_report.json`, `fairness_summary.md`).

## Interpreting results

| Metric | Healthy | Investigate |
|---|---|---|
| Consistency | 100% | any case below 100% |
| Parity gap | < 0.10 | > 0.20 blocks release (`--max-parity-gap`) |
| Sentiment gap | < 0.05 | systematic tone differences per group |
| Stereotype flags | 0 | every flagged output, manually |

A parity gap alone doesn't prove unlawful discrimination — it proves *where to look*. Every flagged case should be reviewed by a human before it becomes a release decision.

## Layout

```
src/counterfactuals.py   prompt variants per demographic group
src/models.py            mock + OpenAI-compatible backends
src/metrics.py           fairness metrics (pure functions)
src/evaluator.py         CLI runner + CI gate
tests/test_fairness.py   unit tests (deterministic, no API)
datasets/                (reserved for larger golden sets)
reports/                 generated per run (gitignored)
```
