## Week 7 Logbook Entry

# Simulation Results: 
https://docs.google.com/spreadsheets/d/1NgcpI-dWA8Xe1BSP5sJrVjH97D_cktXGeLwBL24Ozh8/edit?usp=sharing

### **Discussion Points**  
- Successfully resumed and completed the **30-test kit run** comparing Scenario 4 (Hybrid Extractive-Abstractive) and Scenario 5 (Novelty Model with Sentiment Integration).  
- Continued working on **GitHub Codespace** due to persistent local environment issues, ensuring uninterrupted testing and evaluation.  
- Observed a general **draw between Scenario 4 and Scenario 5**, indicating context-specific effectiveness of each approach.  
- Collected and saved all results for further analysis, including **ROUGE-1, ROUGE-2, ROUGE-L, and Cosine Similarity metrics**.  

### **Work Completed**  
- Conducted the **full 30-test kit run** involving both scenarios, ensuring consistency across evaluations.  
- Exported results to a **CSV file** for further comparison and statistical analysis.  
- Identified a general trend of parity between Scenario 4 and Scenario 5, suggesting both methods are effective but under different conditions.  
- Compiled average ROUGE and cosine similarity scores, noting some variations depending on the dataset.  
- Prepared **comparative graphs and charts** to better visualize the performance of both scenarios.  

### **Performance Summary (Key Observations)**  
- **Scenario 4 vs Source:** ROUGE-L F1 average = 0.2308  
- **Scenario 4 vs Benchmark:** ROUGE-L F1 average = 0.2018  
- **Scenario 5 vs Source:** ROUGE-L F1 average = 0.2233  
- **Scenario 5 vs Benchmark:** ROUGE-L F1 average = 0.2767  

- Results indicate a **general equivalency**, but Scenario 5 tends to perform better with sentiment-heavy datasets.  
- Interesting trend: Scenario 5 shows improved performance when sentiment-heavy datasets are prioritized, while Scenario 4 maintains a stronger performance on factual summaries.  

### **Work Being Completed**  
- Reviewing the results to identify **trends or biases in the scoring system**.  
- Preparing the documentation for the final report, focusing on presenting the draw between the two approaches as a significant finding.  
- Refining the evaluation framework to further explore **qualitative analysis** to supplement quantitative results.  

### **Plan for Week 8**  
- Experiment with **parameter tuning**, particularly the `alpha` value influencing sentiment scoring in Scenario 5.  
- Investigate if combining both methods (Scenario 4 & 5) could yield better results.  
- Continue enhancing the documentation to present findings effectively.  
- Explore the trends to understand the constraints, may it be input or format driven. 
- Explore potential hybridization strategies by selectively incorporating sentiment signals into Scenario 4’s workflow.  

### **Queries for Clarification**  
- What trends can be identified from the average ROUGE scores and cosine similarity comparisons?  
- Should further testing be conducted with different alpha values, and if so, what range is most appropriate?  
- How to effectively communicate a draw in the final thesis, while still highlighting contributions?  
