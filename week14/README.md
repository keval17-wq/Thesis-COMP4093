# Week 14 Logbook Entry

## Discussion Points
- Week 13’s supervisor feedback exposed critical hallucinations in Scenario 5 summaries, particularly under stress conditions (sarcasm, misdirection).
- The thesis' prior assumption — that the system handled these — was invalidated after manual audit.
- Week 14 marked a full pivot: abandoning claims of universal robustness and reframing the project around **faithfulness, sentiment alignment, and compression safety**.
- Final narrative grounded in what the system **reliably** achieves: fact-grounded, sentiment-aware summarisation with strict length constraints.

## Work Completed
- Built and deployed **Scenario 5 v2**, with architecture updated as follows:
  - Sentence-level embeddings using MiniLM
  - KMeans clustering for topic separation
  - Sentiment re-weighting using RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment`)
  - Evidence-first selection followed by length-controlled GPT-4 summarisation
- Re-ran all evaluation sets (Amazon, Random, Stress categories) using Scenario 5 v2.
- Manually reviewed summaries to confirm:
  - ✅ Evidence alignment
  - ✅ Absence of hallucination
  - ✅ Sentiment balance
  - ❌ Failure on linguistic subtleties (sarcasm, misdirection) — acknowledged transparently
- Updated all tables, graphs, and thesis discussions to reflect this corrected scope.

## Key Observations
- Scenario 5 v2 did not solve sarcasm or misdirection, but avoided hallucinating on literal, fact-based reviews.
- Sentiment-weighted selection helped surface minority complaints more consistently.
- Compression logic (via token-length guarding and fallback rephrasing) kept outputs concise and information-rich.
- Evaluation section of thesis was restructured to show stress types were **evaluated**, not necessarily solved.

## Work Being Completed
- Rewriting Chapter 4 and Discussion sections to position Scenario 5 v2 not as “universal,” but as **faithfulness-first**.
- Presentation materials now focused on the shift from metric-trust to manual verification and system safety.

## Plan for Week 15 (Post-Submission)
- Present the project honestly: a controlled, evidence-aligned summariser built to outperform GPT-4 in reliability — not linguistic nuance.
- Highlight:
  - The hallucination failure of G-Eval
  - The value of extractive grounding
  - The system’s readiness for use cases demanding **faithfulness over fluency**

## Queries
- Should G-Eval’s failure and our manual audit process be reported as an insight in its own right?
- Would further work on sarcasm and misdirection require a fundamentally different model (e.g., discourse-aware transformers)?
