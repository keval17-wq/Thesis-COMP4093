# Week 13 Logbook Entry (Critical Turning Point)

## Discussion Points
- Thursday supervisor meeting exposed hallucinations in SC5 summaries, despite high G-Eval scores.
- Previously trusted evaluation (automated) was proven misleading on closer manual inspection.
- Entire thesis narrative — “SC5 beats GPT-4” — was invalidated.

## Work Completed
- Audited all SC5 outputs from stress categories manually.
- Identified fabricated facts and unsupported summaries that passed G-Eval undetected.
- Decided to scrap prior evaluation conclusions and rebuild from zero with a corrected pipeline.

## Key Observations
- G-Eval missed subtle hallucinations in stress test scenarios.
- Faithfulness must be checked manually; metrics are insufficient.
- Need for new evaluation architecture that enforces evidence-first abstraction.

## Work Being Completed
- Prototyping a new version of Scenario 5 (v2):
  - MiniLM embeddings
  - KMeans clustering
  - Sentiment-weighted re-ranking
  - Hard length guard + GPT-4 abstraction

## Plan for Week 14
- Re-run all experiments using Scenario 5 v2.
- Generate new ideal summaries as gold references.
- Frame new thesis narrative: “faithfulness-first summarisation under constraint.”

## Queries
- Should previous findings be explicitly disavowed or just rephrased?
- Is it valid to submit thesis based on Scenario 5 v2 despite late-stage pivot?
