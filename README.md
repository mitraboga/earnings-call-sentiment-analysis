# 📞 S&P 500 Earnings Call Sentiment Analysis (VADER + FinBERT) → Power BI Intelligence Dashboard

Turn raw earnings-call transcripts into **decision-grade sentiment metrics** that quantify how **management, analysts, and the market narrative** evolve over time.

This repository builds an end-to-end NLP pipeline that produces:
- ✅ **Clean speaker-level text blocks**
- ✅ **Speaker role labeling** (Management / Analyst / Operator / Other)
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
Sentiment scoring helps detect changes in:
- **Confidence vs caution** (tone shifting positive → neutral/negative)
- **Uncertainty language** (hedging, vague guidance)
- **Pressure dynamics** (analysts pushing back vs management defending)
- **Narrative momentum** across quarters and companies

### Real-world business advantages
These insights can support:
- **Investor Relations (IR):** refine messaging; identify where investors are unconvinced
- **Equity Research:** add consistent sentiment KPIs to qualitative call notes
- **Risk / Compliance:** flag unusually negative calls for deeper review
- **Portfolio strategy:** compare narrative trend across companies and time
- **Competitive intelligence:** benchmark management confidence vs peers

### What this proves as an NLP / AI / ML engineering project
This repo demonstrates the core value loop:

**Raw text → structured speaker blocks → model scoring → aggregated KPIs → business dashboard**

That’s exactly how NLP engineers turn unstructured language into **measurable metrics** that guide decisions.

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
```
---

## Data Source

This project uses the Hugging Face dataset:
- `kurry/sp500_earnings_transcripts`

`etl/load_transcripts.py` exports it to:
- `data/raw/transcripts_raw.csv`

---

## Setup

### 1) Create and activate a virtual environment (recommended)
Windows:
    python -m venv .venv
    .venv\Scripts\activate

macOS/Linux:
    python -m venv .venv
    source .venv/bin/activate

### 2) Install dependencies
    pip install -r requirements.txt

IMPORTANT: `etl/load_transcripts.py` uses Hugging Face `datasets`:
    from datasets import load_dataset

So install it (and ideally add it to requirements.txt):
    pip install datasets

### 3) Install spaCy model
Your preprocessing uses:
    spacy.load("en_core_web_sm", disable=["parser", "ner"])

Install:
    python -m spacy download en_core_web_sm

---

## Pipeline (Run Order)

### Step 1 — Download transcripts → data/raw/transcripts_raw.csv
    python etl/load_transcripts.py

Output:
- data/raw/transcripts_raw.csv

---

### Step 2 — Preprocess into clean speaker blocks + role inference
    python etl/preprocess_speaker_blocks.py

Outputs:
- data/processed/speaker_blocks_cleaned.csv
- data/processed/preprocess_checkpoint.txt (resume support)

What preprocessing does (technical):
- Reads raw transcripts from `data/raw/transcripts_raw.csv`
- Parses `structured_content` safely via:
  - json.loads() first
  - ast.literal_eval() fallback
- Extracts speaker blocks from common keys (segments/blocks/content/dialogue)
- Cleans text:
  - lowercasing
  - removes “forward-looking statements” and “safe harbor” sections
  - strips non-alphabet characters
  - lemmatizes with spaCy
  - removes stopwords
  - keeps alpha tokens with length > 2
- Filters low-signal blocks using:
  - MIN_BLOCK_LEN = 30 words
- Labels speaker role using keyword heuristics:
  - operator / management / analyst / other

Output schema: data/processed/speaker_blocks_cleaned.csv
- symbol
- company_name
- year
- quarter
- date
- speaker
- speaker_role
- clean_text
- block_length

---

### Step 3 — VADER sentiment scoring (baseline)
    python models/sentiment_vader.py

Outputs:
- data/processed/speaker_blocks_with_vader.csv
- data/processed/vader_checkpoint.txt

Adds:
- sentiment_vader  (compound score in [-1, 1])

Notes:
- Uses chunked processing (CHUNK_ROWS = 5000)
- Resume support via vader_checkpoint.txt
- Duplicate-protection: script stops if output exists and checkpoint is 0

---

### Step 4 — FinBERT sentiment scoring (finance transformer)
    python models/sentiment_finbert.py

Expected output (required for merge step):
- data/processed/speaker_blocks_with_finbert.csv

Required columns (used downstream):
- finbert_sentiment   (positive / neutral / negative)
- finbert_confidence  (confidence score)

IMPORTANT:
- Your merge + aggregation scripts assume the file/columns above exist.
- If your current `sentiment_finbert.py` is not producing them yet, implement/update it so it writes:
  data/processed/speaker_blocks_with_finbert.csv

---

### Step 5 — Merge VADER + FinBERT into one dataset
    python features/merge_sentimnets.py

Inputs:
- data/processed/speaker_blocks_with_vader.csv
- data/processed/speaker_blocks_with_finbert.csv

Output:
- data/processed/speaker_blocks_with_sentiment.csv

Merge method:
- strict inner join on:
  symbol, company_name, year, quarter, date,
  speaker, speaker_role, clean_text, block_length

---

### Step 6 — Build Power BI metric tables (call-level + role-level)
    python features/aggregate_for_powerbi.py

Outputs:
- data/processed/powerbi_call_level_metrics.csv
- data/processed/powerbi_role_level_metrics.csv

FinBERT → numeric mapping (for aggregation and easy charting):
- positive → +1
- neutral  → 0
- negative → -1

Role-level metrics (granularity: call + speaker_role):
- blocks
- avg_block_len
- vader_mean, vader_median
- finbert_mean
- finbert_pos / finbert_neg / finbert_neu (as proportions)
- finbert_avg_conf

Call-level metrics (granularity: call):
- total_blocks
- avg_block_len
- vader_mean
- finbert_mean
- finbert_pos / finbert_neg / finbert_neu
- finbert_avg_conf
- management vs analyst gaps (if both roles exist):
  - vader_gap_mgmt_minus_analyst
  - finbert_gap_mgmt_minus_analyst

---

## Power BI Dashboard (technical build + how it works)

File:
- NLP-Dashboard.pbix

This dashboard is designed to be fast and scalable by using **pre-aggregated CSV outputs** generated in Python
(instead of forcing Power BI to perform heavy NLP / large-group aggregations).

### Data sources used (the dashboard inputs)
1) data/processed/powerbi_call_level_metrics.csv
   - One row per call (symbol + year + quarter + date)

2) data/processed/powerbi_role_level_metrics.csv
   - One row per (call + speaker_role)

### Data model approach (recommended)
In Model view, use a simple relationship:
- powerbi_call_level_metrics (1) → (many) powerbi_role_level_metrics

Join keys (from the pipeline):
- symbol, year, quarter, date
(Optional: company_name, if consistent)

Best practice (cleaner):
- Create a single `call_id` column in Power Query on BOTH tables:
  call_id = symbol & "-" & year & "-" & quarter & "-" & date
- Relate tables on `call_id` instead of multi-column joins

### Power Query (Transform Data) technical steps
Typical steps applied when importing:
- Set data types:
  - date → Date
  - year / quarter → Whole number or Text (depending on visuals)
  - sentiment metrics → Decimal number
- Validate nulls:
  - gap metrics may be blank when analyst or management rows are missing
- Optional (advanced):
  - create a Date table for proper time intelligence (YoY, QoQ)

### DAX measures (lightweight because Python did the heavy lift)
Because metrics are precomputed, DAX can stay clean and readable.

Examples:
- Avg Call VADER:
  Avg Call VADER = AVERAGE(powerbi_call_level_metrics[vader_mean])

- Avg Call FinBERT:
  Avg Call FinBERT = AVERAGE(powerbi_call_level_metrics[finbert_mean])

- Avg Mgmt vs Analyst Gap (VADER):
  Avg VADER Gap (Mgmt-Analyst) = AVERAGE(powerbi_call_level_metrics[vader_gap_mgmt_minus_analyst])

- Avg % Positive (FinBERT):
  Avg FinBERT % Positive = AVERAGE(powerbi_call_level_metrics[finbert_pos])

- Total Blocks:
  Total Blocks = SUM(powerbi_role_level_metrics[blocks])

### What the dashboard enables (analysis patterns)
With these two tables, the dashboard can support:
- KPI cards (avg sentiment, proportions, confidence, volume)
- time trends (sentiment over time by company/quarter)
- role comparisons (management vs analyst vs other)
- distribution visuals (FinBERT positive/neutral/negative)
- gap analysis (mgmt-minus-analyst divergence per call)

### Refresh workflow
1) Run the Python pipeline to regenerate CSVs in data/processed/
2) Open NLP-Dashboard.pbix
3) Power BI → Transform Data → Data Source Settings
4) Confirm file paths point to your local data/processed/ folder
5) Refresh

---

## Checkpointing (resume support)

Built for long runs + safe interruption:
- data/processed/preprocess_checkpoint.txt
- data/processed/vader_checkpoint.txt

If interrupted (CTRL+C), rerun the script to resume from the last checkpoint.

---

## Troubleshooting

spaCy model missing:
- Error: Can't find model 'en_core_web_sm'
- Fix:
    python -m spacy download en_core_web_sm

Hugging Face datasets missing:
- Error: No module named 'datasets'
- Fix:
    pip install datasets

VADER duplicate protection:
- If you want a clean rebuild, delete:
  - data/processed/speaker_blocks_with_vader.csv
  - data/processed/vader_checkpoint.txt

---

## Author
Mitra Boga
