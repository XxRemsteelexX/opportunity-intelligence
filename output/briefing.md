# Opportunity Intelligence Briefing
## Des Moines Metro (Polk County, IA)
**Generated**: 2026-02-14 22:50

---

**Executive Briefing – Des Moines Metro (Polk County, IA) – Potential Senior Living Community**

---

### 1. Market Overview  
Polk County’s total population is 492,401, with 59,900 residents age 65+ (12.2% of the market) and 6,876 adults 85+ who drive the highest‑acuity care demand【Census-1】【Census-3】. Median household income stands at $72,562, indicating solid spending power for senior‑care services【Census-4】.

### 2. Demand Indicators  
| Indicator | Value | Implication |
|-----------|-------|-------------|
| Seniors (65+) | 59,900 | Base pool for independent, assisted, and skilled‑care units【Census-2】 |
| 85+ cohort | 6,876 | Concentrated need for higher‑level nursing and memory‑care beds【Census-3】 |
| Seniors in poverty | 5,169 (≈8.6% of 65+) | Potential market for subsidized or Medicaid‑eligible beds【Census-7】 |
| Housing vacancy | 6.0% | Sufficient available land/units for new construction or conversion【Census-6】 |
| Beds per 1,000 seniors | 15.0 | Slightly above the national average (~13) but still leaves room for growth【CMS-3】 |
| Occupancy rate (existing) | 83.4% | 16.6% of beds are vacant, indicating capacity to absorb new entrants【CMS-4】 |

### 3. Competitive Landscape  
*Supply*: 10 nursing facilities provide 896 certified beds, averaging 77 beds per provider【CMS-1】【CMS-2】. The market is **unconcentrated** (HHI = 1,149) with the top three operators holding 44.6% of capacity【Stat-8】【Stat-9】【Stat-10】.  

*Quality*: 60% of facilities earn 4‑5 stars; the remaining 40% are 1‑3 stars, with two 2‑star for‑profit homes (Sunny View, Kennybrook) pulling occupancy below 70%【CMS-5】【CMS-6】【Fac-5】【Fac-9】. The average overall rating is 3.7/5, and a strong positive correlation exists between overall rating and staffing rating (r = 0.822)【Stat-5】, suggesting staffing quality drives performance.  

*Ownership*: Non‑profits control 60% of market share and dominate the high‑rating tier (100% of non‑profit rows are 4‑5 stars)【CMS-7】【Stat-24】. A chi‑square test confirms a significant link between ownership type and rating tier (χ² = 10.00, p = 0.0067)【Stat-16】【Stat-25】.  

*Performance Gaps*: Facilities with lower ratings (2‑star) have occupancy rates 12‑20 points below the market average (e.g., Sunny View 72.5%, Kennybrook 68.9%)【Fac-5】【Fac-9】. High‑rated homes (5‑star) consistently exceed 85% occupancy, some reaching 94% (Edgewater)【Fac-3】【Fac-8】. This gap signals an opportunity for a well‑staffed, high‑quality community to capture market share.

### 4. Risks and Considerations  
1. **Moderate Demand Pressure** – The market scoring service rates demand pressure at 53.6/100, indicating demand is present but not overwhelming; sub‑market analysis (e.g., specific zip codes, senior income brackets) is needed before committing【Score-2】.  
2. **Competitive Saturation in Certain Segments** – While overall concentration is low, the top three providers already hold nearly half the beds, which could limit rapid entry in the mid‑range assisted‑living segment【Stat-10】.  
3. **Ownership‑Rating Dynamics** – Non‑profit operators dominate high‑rating tiers; a for‑profit entrant may face perception challenges unless it can demonstrably exceed staffing and quality benchmarks【Stat-16】.  
4. **Regulatory & Staffing Constraints** – High multicollinearity among rating, occupancy, health‑inspection, and staffing variables (VIF > 10) suggests these factors move together; any shortfall in staffing could quickly depress ratings and occupancy【Stat-12】.  
5. **Economic Sensitivity** – Although median income is strong, 8.6% of seniors are in poverty, limiting the pool for premium pricing; a mixed‑payer model (private pay + Medicaid) may be required【Census-4】【Census-7】.

### 5. Recommendation  
**Go‑ahead with a “needs‑more‑research” stance**. The overall opportunity score is 48.3/100—below the midpoint—reflecting modest upside【Score-1】. However, the clear quality gap (low‑rated facilities under‑performing on occupancy) and the un‑concentrated market (HHI = 1,149) provide a viable entry point for a high‑quality, well‑staffed community, especially targeting the 85+ cohort.  

**Next steps**:  
- Conduct sub‑market feasibility (zip‑code level senior density, income, and Medicaid eligibility).  
- Model a mixed‑payer pricing structure to address the 8.6% senior poverty rate.  
- Develop a staffing plan that meets or exceeds the high‑rating correlation threshold (staffing rating > 4.5) to protect occupancy performance.  

If sub‑market analysis confirms sufficient unmet demand and favorable payer mix, proceed to site acquisition and design. Otherwise, defer until market conditions improve.  

---

## Source Index

| Tag | Source | Description |
|-----|--------|-------------|
| Census-1 through Census-7 | US Census Bureau ACS 5-Year (2022) | Demographics, income, housing for Polk County, IA |
| CMS-1 through CMS-8 | CMS Care Compare Provider Data | Nursing home ratings, beds, occupancy |
| Stat-1 through Stat-30 | Statistical Analysis Service | LLM-directed analyses from analytics library |
| Score-1 through Score-4 | Market Scoring Service | Demand pressure, competitive position, opportunity scoring |
| Fac-1 through Fac-10 | Fac | Derived |

---

## Instrumentation

| Metric | Value |
|--------|-------|
| cms_api_fetch | 0.63s |
| llm_analysis_planning | 2.08s |
| llm_followup_review | 2.72s |
| llm_final_briefing | 4.75s |
| Total tokens | 7,457 |
| Prompt tokens | 3,951 |
| Completion tokens | 3,506 |
| Estimated cost | $0.0644 |
| Model | openai/gpt-oss-120b |

## Analysis Pipeline Summary

| Step | Service | Detail |
|------|---------|--------|
| Data Profiling | TypeInferenceService, ModelRecommender | 8 columns profiled, 8 analyses recommended |
| Market Scoring | MarketScorer | Demand=53.6, Competitive=40.3, Overall=48.3 |
| LLM Analysis Planning | openai/gpt-oss-120b | Selected 9 analyses from library |
| Statistical Analysis | StatisticalAnalyzer, AnomalyDetector | 14 analyses executed |
| LLM Follow-up Review | openai/gpt-oss-120b | Requested 5 additional |
| LLM Synthesis | openai/gpt-oss-120b | Final executive briefing with citations |

### Services Used
| File | Service | Usage |
|------|---------|-------|
| `src/type_inference.py` | TypeInferenceService | Column type detection, profiling |
| `src/statistical.py` | StatisticalAnalyzer | 13 statistical analyses |
| `src/anomaly_detection.py` | AnomalyDetector | Z-score and isolation forest detection |
| `src/market_scoring.py` | MarketScorer | Demand pressure + competitive position scoring |
| `src/report_builder.py` | EvidenceBuilder, ReportCompiler | Evidence packaging + report compilation |
| `src/model_recommender.py` | ModelRecommender | Analysis type recommendation |
| `src/data_cleaning.py` | DataCleaningService | File loading + type casting (used internally by StatisticalAnalyzer) |
