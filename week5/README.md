## Week 5 Logbook Entry
**Discussion Points**  
- Implemented benchmark summaries for evaluation.  
- Integrated ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L) and cosine similarity to assess summary quality.  
- Conducted early comparative tests between Scenario 4 (hybrid extractive-abstractive) and Scenario 5 (novelty model with sentiment integration).

**Work Completed**  
- Finalized the evaluation pipeline to compare generated summaries against benchmark and source texts.  
- Applied ROUGE and cosine similarity metrics to validate summary quality.  
- Observed that Scenario 5 (with novelty and sentiment-aware extraction) outperformed Scenario 4 in key metrics.

**Work Being Completed**  
- Refining logging and output formats for clear comparative analysis.  
- Continuing to adjust parameters (e.g., `top_n`, sentiment weight `alpha`) for optimal performance.

**Plan for Week 6**  
- (Future work) Extend evaluation with CSV export and detailed comparative analysis.  
- Prepare documentation for final evaluation and reporting of the winning approach.

**Queries for Clarification**  
- How to best balance sentiment influence without overshadowing factual content?  
- Should all experimental scenarios be included in the final thesis report, or only the best-performing model?