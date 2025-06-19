# Week 11 Logbook Entry

## Discussion Points
- Supervisor feedback (May 7) urged re-framing results:
  - “How much better is your system?”, not how much worse GPT is.
  - Focus on surfacing “important information”, not just “rare mentions”.
  - Real reviews preferred; synthetic only if justified.
- Proposed robustness test for Scenario 5 summaries — pending methodological soundness.

## Work Completed
- Rewrote evaluation framing in thesis to position SC5 as value-adding.
- Built initial script for BERTScore, but did not run due to pipeline error with Scenario 5 v2.
- Pulled real Amazon review sets and confirmed their use in Phase 2 experiments.
- Drafted score tables showing SC5’s factual, sentiment, and length improvements.

## Key Observations
- SC5 retained complaint diversity and sentiment integrity better than vanilla GPT-4.
- G-Eval + ROUGE scores matched expectations, but only measured surface-level fluency.

## Work Being Completed
- Rebuilding per-category metric summaries and preparing comparative plots.
- Investigating the BERTScore error (possibly tensor shape or tokenizer mismatch).

## Plan for Week 12
- Finalise full system comparison section.
- Address feedback on evaluation logic.
- Complete all experiment reruns and thesis draft integration.

## Queries
- Is BERTScore necessary if G-Eval and manual review already align?
- Should important-fact coverage be manually tagged?
