## Week 8 Logbook Entry

### **Discussion Points**
- Began assembling summary sample sets for **blind human evaluation** — ensuring reviewers couldn’t infer authorship or scenario.
- Took sample summaries: **Sum 2, 6, 9, 16, 20, 21** and shuffled order for unbiased comparison.
- Supervisor review reinforced focus on **summary size**, diversity of human judges, and eventual **BERTScore comparison** to match ROUGE.

### **Work Completed**
- Developed first round of anonymised forms and sample evaluations (Doc D).
- Trialled first batch of human review internally and tracked feedback.
- Explored using BERTScore for semantic-level verification.
- Continued investigating `.env` and deployment blocking issues for Streamlit.

### **Key Observations**
- Human ratings exposed gaps that ROUGE could not reliably detect.
- Evaluation required **different annotators** to rule out confirmation bias.
- BERTScore appears more aligned with nuanced summary preferences.

### **Work Being Completed**
- Aggregating scores into the revised evaluation table.
- Comparing BERTScore, ROUGE, and human ranking outputs for agreement.
- Preparing documentation for evaluation workflow.

### **Plan for Week 9**
- Complete ideal summaries for all test sets.
- Re-run system summaries against those references with all metrics applied.
- Restart Streamlit deploy with simplified `.env` and access control.

### **Queries**
- Should human evaluators be credited or anonymised in the appendix?
- Can human vs. BERTScore disagreement be framed as a robustness test?
- How many samples are statistically “enough” for subjective eval?
