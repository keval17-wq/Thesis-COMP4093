# Week 10 Logbook Entry (Starting 12 May 2025)

## Discussion Points
- Began final evaluation phase for ESA pipeline, focusing on SC5 (Scenario 5 with sentiment weighting).
- Maintained narrative that SC5 outperformed vanilla GPT-4 across Phase 1–3 datasets.
- Confirmed all summaries were shorter than source inputs, per thesis design constraints.

## Work Completed
- Ran SC5 and vanilla GPT-4 on Amazon sets and synthetic stress categories.
- Integrated ideal summaries and recalculated ROUGE and BERTScore (partial).
- Updated evaluation harness to report summary/input length ratios and rare-issue inclusion.

## Key Observations
- SC5 summaries were shorter, denser, and more sentiment-preserving.
- Length constraints functioned as intended, with minor need for truncation.
- G-Eval results strongly favoured SC5 for coherence and relevance.

## Work Being Completed
- Finalising all table outputs and score logs for thesis.
- Formatting figures and writing interpretation sections for Chapter 4.

## Plan for Week 11
- Begin integrating these insights into final thesis results.
- Draft conclusions based on observed superiority of SC5.
- Prepare for supervisor feedback session.

## Queries
- Is it sufficient to focus on G-Eval/ROUGE/BERTScore, or is human rating still required?
- Should summary examples be included inline or pushed to appendix?
