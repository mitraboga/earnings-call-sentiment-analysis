import pandas as pd
from pathlib import Path

VADER_PATH = Path("data/processed/speaker_blocks_with_vader.csv")
FINBERT_PATH = Path("data/processed/speaker_blocks_with_finbert.csv")
OUT_PATH = Path("data/processed/speaker_blocks_with_sentiment.csv")

# Join keys (must exist in BOTH files)
KEYS = [
    "symbol", "company_name", "year", "quarter", "date",
    "speaker", "speaker_role", "clean_text", "block_length"
]

if __name__ == "__main__":
    if not VADER_PATH.exists():
        raise FileNotFoundError(f"Missing {VADER_PATH}")
    if not FINBERT_PATH.exists():
        raise FileNotFoundError(f"Missing {FINBERT_PATH}")

    vader = pd.read_csv(VADER_PATH)
    finbert = pd.read_csv(FINBERT_PATH)

    # Keep only needed columns from finbert
    finbert = finbert[KEYS + ["finbert_sentiment", "finbert_confidence"]]

    merged = vader.merge(finbert, on=KEYS, how="inner")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print("VADER rows:", vader.shape)
    print("FinBERT rows:", finbert.shape)
    print("Merged rows:", merged.shape)
    print("Saved ->", OUT_PATH.resolve())