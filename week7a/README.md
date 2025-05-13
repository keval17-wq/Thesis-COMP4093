## Week 7a Logbook Entry (Break Week 1)

### **Discussion Points**
- Introduced a **length guard** to ensure summary outputs remain meaningfully **shorter than original inputs**.
- Faced difficulties in initial enforcement — early outputs trimmed too harshly, reducing key sentiment-bearing details.
- Balance between compression and informativeness became a core constraint to resolve.

### **Work Completed**
- Ran multiple prompt iterations embedding token-length guidance.
- Added semantic backtracking steps to verify **coverage was not lost**.
- Reviewed and manually rated summary density vs readability trade-offs.

### **Key Observations**
- Too tight a bound led to shallow outputs; too loose reintroduced redundancy.
- **Quote-style anchoring** helped stabilize outputs under length constraints.
- Subtle refinements in prompt framing offered higher-quality outputs without overshooting.

### **Work Being Completed**
- Standardising a **length-safe summarisation recipe** for Scenario 5.
- Preparing 10–15 ideal references to compare future generations against.
- Started testing summary/input ratio thresholds across the set.

### **Plan for Week 8**
- Merge length constraint into main evaluation harness.
- Re-evaluate old summaries with new constraint applied.
- Continue addressing Streamlit deployment blocking via `.env` adjustments.

### **Queries**
- Should “succinctness” be reported as a derived metric?
- What’s the tradeoff between extractive density vs abstraction quality?
- Can partial sentence completions help mitigate length overshoot?
