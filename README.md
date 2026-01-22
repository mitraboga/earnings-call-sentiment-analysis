# 📞 S&P 500 Earnings Call Sentiment Analysis (VADER + FinBERT) → Power BI Intelligence Dashboard

Turn raw earnings-call transcripts into **decision-grade sentiment metrics** that quantify how **management, analysts, and the market narrative** evolve over time.

This repository builds an end-to-end NLP pipeline that produces:
- ✅ **Clean speaker-level text blocks**
- ✅ **Speaker role labelling** (Management / Analyst / Operator / Other)
- ✅ **Dual-model sentiment scoring**
  - **VADER** (fast, rule-based baseline)
  - **FinBERT** (finance-domain transformer)
- ✅ **Power BI-ready metrics tables**
- ✅ A Power BI dashboard: `NLP-Dashboard.pbix`

---

## Why this matters (real business impact)

Earnings calls are not “just text.” They are strategic communication events that influence:
- **Investor confidence**
- **Risk perception**
- **Market narratives**
- **Competitive positioning**

### What sentiment analysis unlocks
Sentiment scoring helps teams detect changes in:
- **Confidence vs caution** (tone shifting positive → neutral/negative)
- **Uncertainty language** (hedging, vague forward guidance)
- **Pressure dynamics** (analysts pushing back vs management defending)
- **Narrative momentum** across quarters and companies

### Real-world business advantages
These insights can directly support:
- **Investor Relations (IR):** improve messaging; identify where investors are unconvinced
- **Equity Research:** add consistent, quantifiable tone metrics to qualitative call notes
- **Risk / Compliance:** flag unusually negative calls for deeper review
- **Portfolio strategy:** compare “narrative trend” across companies and time
- **Competitive intelligence:** benchmark management confidence against peers

### What this proves as an NLP / AI / ML engineering project
This repo demonstrates a high-value engineering loop:

**Raw text → structured speaker blocks → model scoring → aggregated KPIs → business dashboard**

That’s exactly how NLP engineers convert unstructured language into:
- structured datasets,
- measurable metrics,
- and analytics assets that inform decisions.

---

## Repo Structure (matches your actual repo)

```text
.
├── data/
│   ├── raw/
│   │   └── transcripts_raw.csv
│   └── processed/
│       ├── speaker_blocks_cleaned.csv
│       ├── speaker_blocks_with_vader.csv
│       ├── speaker_blocks_with_finbert.csv          # expected output for merge
│       ├── speaker_blocks_with_sentiment.csv
│       ├── powerbi_call_level_metrics.csv
│       ├── powerbi_role_level_metrics.csv
│       ├── preprocess_checkpoint.txt
│       └── vader_checkpoint.txt
│
├── etl/
│   ├── load_transcripts.py
│   └── preprocess_speaker_blocks.py
│
├── models/
│   ├── sentiment_vader.py
│   └── sentiment_finbert.py
│
├── features/
│   ├── merge_sentimnets.py              # filename typo is intentional (matches repo)
│   └── aggregate_for_powerbi.py
│
├── NLP-Dashboard.pbix
└── requirements.txt
