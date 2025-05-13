## Week 9 Logbook Entry

### **Discussion Points**
- Supervisor feedback (Apr 16) stressed importance of finalising **ideal summaries** for the dataset.
- Re-confirmed that summaries must remain **within defined size boundaries**, always shorter than inputs.
- Length and alignment now core factors in model validation.

### **Work Completed**
- Created and reviewed ideal reference summaries.
- Integrated those into the evaluation comparison pipeline.
- Verified model summaries for compliance on length constraint.
- Completed a full metric sweep: ROUGE, BERTScore, and ratio checks.

### **Key Observations**
- Subtle patterns emerged — **Scenario 5 aligned most closely** to reference summaries across multiple metrics.
- These results weren’t extreme, but consistently tipped in favour of the sentiment-conditioned layer.
- Ground truth alignment improved when **quote relevance and brevity intersected**.

### **Work Being Completed**
- Prepping illustrative examples for the report appendix.
- Resolving any residual metric mismatch on updated references.
- Revalidating human score sheets against the new outputs.

### **Plan for Week 10**
- Complete and finalise **Streamlit deployment**.
- Freeze all outputs and logs.
- Align documentation with the evolving evaluation insights.

### **Queries**
- Can we include summary length statistics to justify constraints?
- Should sentiment presence be flagged as an input feature going forward?
- Is it too assertive to say Scenario 5 is “better,” or is “aligned more reliably” sufficient?
