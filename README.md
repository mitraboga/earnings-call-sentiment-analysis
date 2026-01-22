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

---

## 📊 Power BI Dashboard — *Earnings Call Sentiment Analysis (Executive Overview)*

This project isn’t just NLP in Python — the **Power BI layer is where raw text becomes decisions**.

The dashboard turns thousands of earnings-call speaker blocks into **executive-ready KPIs**, letting you:
- spot **sentiment shifts** over time (quarter-to-quarter),
- compare **Management vs Analyst tone** (credibility vs skepticism),
- separate **Prepared Remarks vs Q&A** (scripted vs spontaneous),
- and drill into **outliers + confidence** using purpose-built tooltips.

> **Business impact (real world):**  
> Sentiment signals can act like an “early-warning system” for guidance risk, investor expectations, PR issues, or competitive pressure — especially when **Management’s tone diverges from Analysts’ tone**.

---

## 🧠 What the Dashboard Measures (in plain English)

**Two sentiment engines, two perspectives:**
- **VADER (lexicon-based):** fast, rule-based sentiment score (range: **-1 to +1**) on cleaned text blocks.
- **FinBERT (finance-tuned transformer):** classifies sentiment (**positive / neutral / negative**) with a **confidence score**.

**Core metrics surfaced in Power BI:**
- **Avg FinBERT** (call-level mean score derived from FinBERT labels)
- **Avg VADER** (call-level mean compound)
- **Avg Confidence** (mean FinBERT confidence)
- **Sentiment Mix** (% Positive / Neutral / Negative)
- **Role Gap** (**Management − Analyst**) using both VADER + FinBERT
- **QoQ Δ** (quarter-over-quarter change for sentiment)

---

## 🧩 Data Model + How It’s Wired

This dashboard is backed by the pipeline outputs generated from Python:

- `data/processed/powerbi_call_level_metrics.csv`  
  **One row per earnings call** (company + quarter + date), with aggregate sentiment metrics.

- `data/processed/powerbi_role_level_metrics.csv`  
  **One row per call + role (+ section)** with role-specific aggregates (Management / Analyst / Operator).

**Model design highlights:**
- Two fact tables (**Call-level** + **Role-level**) connected via a shared **CallKey** logic (and aligned on call dimensions like symbol/year/quarter/date).
- A small `DimSection` table (Prepared Remarks vs Q&A) enabling clean section filtering.
- Dedicated tooltip helper tables (`TT_*`) + a measures table enabling:
  - dynamic tooltip titles,
  - confidence badges,
  - context-aware drilldowns without cluttering the main page.

---

## 🖥️ Report Page Walkthrough (What each visual answers)

### 1) KPI Banner (Top)
Quick “executive snapshot” of the current filter context:
- **Total Calls**
- **Avg FinBERT**
- **Avg VADER**
- **Avg Confidence**

### 2) Earnings Call Sentiment Trend (Line)
Answers: **“Is sentiment improving or deteriorating over time?”**  
Plots FinBERT + VADER trends across quarters and lets users hover for deeper breakdowns.

### 3) Management vs Analyst Sentiment (Bar)
Answers: **“Who’s more optimistic — leadership or the Street?”**  
This is the **credibility / tension signal**. A widening gap can indicate:
- management optimism not matched by analyst belief,
- increased skepticism during Q&A,
- “tone-management” vs real fundamentals.

### 4) Prepared Remarks vs Q&A (Comparison)
Answers: **“Does sentiment drop when the script ends?”**  
Prepared remarks are controlled; Q&A is where risk shows up.

### 5) Sentiment Distribution (Donut)
Answers: **“What’s the overall tone mix?”**  
Shows the % split of positive/neutral/negative sentiment under the current slicers.

---

## 🧠 Tooltips (Deep Insights Without Cluttering the Dashboard)

Power BI tooltips are the “secret weapon” in this report:  
you keep the main dashboard clean, while hover interactions reveal the real diagnostics.

### ✅ Tooltip #1 — Role Gap + Percentile
Shows:
- **FinBERT Gap** and **VADER Gap** (Management − Analyst)
- **Gap Percentile Label** (how extreme the gap is vs historical distribution)
- Role-level matrix with:
  - Sentiment (FinBERT & VADER)
  - Delta metrics (contextual shifts)
- Confidence badge (**High / Medium / Low**) based on FinBERT confidence.

**Why it matters:**  
A gap with **high confidence** is actionable — it means the divergence is consistent across blocks, not noise.

### ✅ Tooltip #2 — Sentiment Mix + Baseline Delta
Shows:
- % **Positive / Neutral / Negative**
- “**vs Company Average**” uplift (contextual benchmark)
- A baseline delta chart showing how the selected context differs from normal.

**Why it matters:**  
This lets you separate:
- “This quarter is positive”  
from  
- “This quarter is positive **relative to what’s typical for this company**.”

### ✅ Tooltip #3 — QoQ Δ + Extremes (Best/Worst Calls)
Shows:
- **FinBERT QoQ Δ** and **VADER QoQ Δ**
- Volume context (**# blocks**, **# calls**)
- **Sentiment Mix**
- **Best / Worst CallKeys** within the quarter (outlier detection)
- Confidence badge (High/Medium/Low)

**Why it matters:**  
This instantly surfaces “what changed” *and* “which calls drove it.”

---

## 🖼️ How to Add Your Screenshots into this README (Copy/Paste)

### Step 1 — Create a folder in your repo
Create:

`assets/powerbi/`

Put your screenshots inside it and rename them like this (recommended):

- `assets/powerbi/dashboard_overview.png`
- `assets/powerbi/table_call_level.png`
- `assets/powerbi/table_role_level.png`
- `assets/powerbi/model_view.png`

Tooltips (full pages):
- `assets/powerbi/tooltip_1_role_gap.png`
- `assets/powerbi/tooltip_2_baseline.png`
- `assets/powerbi/tooltip_3_extremes.png`

Tooltip interactions (the 3 screenshots you said you’ll use in the README):
- `assets/powerbi/tooltip_interaction_baseline.png`
- `assets/powerbi/tooltip_interaction_role_gap.png`
- `assets/powerbi/tooltip_interaction_extremes.png`

### Step 2 — Paste these image embeds

#### Executive Overview (Report View)
<img src="assets/powerbi/dashboard_overview.png" width="1000" />

#### Data Tables (Call-Level + Role-Level)
<details>
  <summary><b>Click to expand (Table View + Model View)</b></summary>
  <br/>

  <b>Call-Level Metrics Table</b><br/>
  <img src="assets/powerbi/table_call_level.png" width="1000" />
  <br/><br/>

  <b>Role-Level Metrics Table</b><br/>
  <img src="assets/powerbi/table_role_level.png" width="1000" />
  <br/><br/>

  <b>Model View (Relationships + Tooltip Helper Tables)</b><br/>
  <img src="assets/powerbi/model_view.png" width="1000" />
</details>

#### Tooltips (Full Pages)
<details>
  <summary><b>Click to expand (Tooltip Pages)</b></summary>
  <br/>

  <b>Tooltip #1 — Role Gap + Percentile</b><br/>
  <img src="assets/powerbi/tooltip_1_role_gap.png" width="1000" />
  <br/><br/>

  <b>Tooltip #2 — Mix + Baseline Delta</b><br/>
  <img src="assets/powerbi/tooltip_2_baseline.png" width="1000" />
  <br/><br/>

  <b>Tooltip #3 — QoQ Δ + Extremes</b><br/>
  <img src="assets/powerbi/tooltip_3_extremes.png" width="1000" />
</details>

#### Tooltips in Action (Hover Interactions)
These are the “money shots” — they prove the dashboard is interactive and insight-dense.

<p>
  <img src="assets/powerbi/tooltip_interaction_role_gap.png" width="320" />
  <img src="assets/powerbi/tooltip_interaction_baseline.png" width="320" />
  <img src="assets/powerbi/tooltip_interaction_extremes.png" width="320" />
</p>

---

## ✅ How to Use the Dashboard (1-minute guide)

1. Pick a **Company** (or compare multiple).
2. Filter by **Year / Quarter**.
3. Toggle **Prepared Remarks vs Q&A** for “scripted vs real”.
4. Hover any key visual to open **tooltips**:
   - Role Gap diagnostics
   - Baseline deltas vs company norm
   - QoQ change + best/worst calls
5. Use the tooltip insights to answer:
   - “Is sentiment moving?”
   - “Is leadership credibility aligned with analysts?”
   - “Where are the outliers and how confident are we?”

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
