import pandas as pd
import sys
import re
import json
import ast
from pathlib import Path
import spacy

RAW_PATH = Path("data/raw/transcripts_raw.csv")
OUT_PATH = Path("data/processed/speaker_blocks_cleaned.csv")
CHECKPOINT_PATH = Path("data/processed/preprocess_checkpoint.txt")

CHUNK_SIZE = 25
MIN_BLOCK_LEN = 30

MGMT_KEYWORDS = [
    "ceo", "cfo", "chief", "president", "coo", "cto",
    "chairman", "chair", "vp", "vice president", "svp", "evp"
]
ANALYST_KEYWORDS = [
    "analyst", "research", "securities", "capital", "bank",
    "goldman", "morgan", "barclays", "jp morgan", "citi", "ubs"
]
OPERATOR_KEYWORDS = ["operator", "moderator", "coordinator"]

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# ---------------- helpers ----------------
def safe_parse(x):
    if pd.isna(x):
        return None
    if isinstance(x, (list, dict)):
        return x
    try:
        return json.loads(x)
    except Exception:
        try:
            return ast.literal_eval(x)
        except Exception:
            return None

def infer_role(speaker):
    s = (speaker or "").lower()
    if any(k in s for k in OPERATOR_KEYWORDS):
        return "operator"
    if any(k in s for k in MGMT_KEYWORDS):
        return "management"
    if any(k in s for k in ANALYST_KEYWORDS):
        return "analyst"
    return "other"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"forward[- ]looking statements.*", " ", text, flags=re.I)
    text = re.sub(r"safe harbor.*", " ", text, flags=re.I)
    text = re.sub(r"[^a-z\s]", " ", text)
    doc = nlp(text)
    return " ".join(
        tok.lemma_ for tok in doc
        if tok.is_alpha and not tok.is_stop and len(tok) > 2
    )

def extract_blocks(structured):
    if not structured:
        return []
    if isinstance(structured, dict):
        for k in ["segments", "blocks", "content", "dialogue"]:
            if k in structured and isinstance(structured[k], list):
                structured = structured[k]
                break
    if not isinstance(structured, list):
        return []

    blocks = []
    for item in structured:
        if not isinstance(item, dict):
            continue
        speaker = item.get("speaker") or item.get("name") or "Unknown"
        text = item.get("text") or item.get("content")
        if isinstance(text, str) and text.strip():
            blocks.append((speaker, text))
    return blocks

# ---------------- main ----------------
if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    calls = pd.read_csv(RAW_PATH)

    # Load checkpoint
    start_idx = 0
    if CHECKPOINT_PATH.exists():
        start_idx = int(CHECKPOINT_PATH.read_text().strip())
        print(f"🔁 Resuming from transcript index {start_idx}")

    file_exists = OUT_PATH.exists()

try:
    for i in range(start_idx, len(calls), CHUNK_SIZE):
        chunk = calls.iloc[i:i + CHUNK_SIZE]
        rows = []

        for _, r in chunk.iterrows():
            structured = safe_parse(r["structured_content"])
            blocks = extract_blocks(structured)

            for speaker, text in blocks:
                cleaned = clean_text(text)
                if len(cleaned.split()) < MIN_BLOCK_LEN:
                    continue

                rows.append({
                    "symbol": r["symbol"],
                    "company_name": r["company_name"],
                    "year": r["year"],
                    "quarter": r["quarter"],
                    "date": r["date"],
                    "speaker": speaker,
                    "speaker_role": infer_role(speaker),
                    "clean_text": cleaned,
                    "block_length": len(cleaned.split())
                })

        if rows:
            pd.DataFrame(rows).to_csv(
                OUT_PATH,
                mode="a",
                header=not file_exists,
                index=False
            )
            file_exists = True

        # Save checkpoint AFTER successfully finishing this chunk
        CHECKPOINT_PATH.write_text(str(i + CHUNK_SIZE))
        print(f"✅ Processed {min(i + CHUNK_SIZE, len(calls))}/{len(calls)} transcripts")

except KeyboardInterrupt:
    # Clean, expected exit
    last = CHECKPOINT_PATH.read_text().strip() if CHECKPOINT_PATH.exists() else str(start_idx)
    print("\n🛑 Stopped by user (CTRL+C).")
    print(f"✅ Progress saved. Next run will resume from transcript index {last}.")
    sys.exit(0)