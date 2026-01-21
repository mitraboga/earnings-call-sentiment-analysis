import pandas as pd
from pathlib import Path
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

IN_PATH = Path("data/processed/speaker_blocks_cleaned.csv")
OUT_PATH = Path("data/processed/speaker_blocks_with_vader.csv")
CHECKPOINT_PATH = Path("data/processed/vader_checkpoint.txt")

CHUNK_ROWS = 5000

nltk.download("vader_lexicon", quiet=True)
sid = SentimentIntensityAnalyzer()

def vader_score(text: str) -> float:
    return sid.polarity_scores(text)["compound"]

if __name__ == "__main__":
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}. Run preprocessing first.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    start_row = 0
    if CHECKPOINT_PATH.exists():
        val = CHECKPOINT_PATH.read_text().strip()
        if val.isdigit():
            start_row = int(val)

    # If resuming, ensure output exists; if not, start fresh
    if start_row > 0 and not OUT_PATH.exists():
        print("⚠️ Checkpoint exists but output file missing. Restarting from 0.")
        start_row = 0

    if start_row == 0 and OUT_PATH.exists():
        # Safety: avoid accidental duplicates
        print("⚠️ Output file already exists and checkpoint is 0.")
        print("   Delete data/processed/speaker_blocks_with_vader.csv if you want a clean rebuild.")
        print("   OR delete data/processed/vader_checkpoint.txt to force rebuild.")
        # We'll still proceed by appending, but that can duplicate.
        # Safer: stop here.
        raise SystemExit("Stopping to prevent duplicate append. Clean the output/checkpoint and rerun.")

    print(f"▶️ VADER starting from row {start_row}")

    # Stream input, but skip already processed rows
    total_processed = start_row
    first_write = not OUT_PATH.exists()

    reader = pd.read_csv(IN_PATH, chunksize=CHUNK_ROWS)

    # Skip chunks until reaching start_row
    rows_to_skip = start_row
    while rows_to_skip > 0:
        try:
            chunk = next(reader)
        except StopIteration:
            print("✅ Nothing left to process.")
            raise SystemExit(0)

        if len(chunk) <= rows_to_skip:
            rows_to_skip -= len(chunk)
            continue
        else:
            # Partial skip inside this chunk
            chunk = chunk.iloc[rows_to_skip:].copy()
            rows_to_skip = 0

            # Process this partial chunk and then continue normally
            chunk["clean_text"] = chunk["clean_text"].fillna("").astype(str)
            chunk["sentiment_vader"] = chunk["clean_text"].apply(
                lambda t: vader_score(t) if t.strip() else 0.0
            )

            chunk.to_csv(OUT_PATH, mode="a", header=first_write, index=False)
            first_write = False
            total_processed += len(chunk)

            CHECKPOINT_PATH.write_text(str(total_processed))
            print(f"✅ VADER processed rows: {total_processed}")

    # Process remaining chunks
    for chunk in reader:
        chunk = chunk.copy()
        chunk["clean_text"] = chunk["clean_text"].fillna("").astype(str)
        chunk["sentiment_vader"] = chunk["clean_text"].apply(
            lambda t: vader_score(t) if t.strip() else 0.0
        )

        chunk.to_csv(OUT_PATH, mode="a", header=first_write, index=False)
        first_write = False
        total_processed += len(chunk)

        CHECKPOINT_PATH.write_text(str(total_processed))
        print(f"✅ VADER processed rows: {total_processed}")

    print("🎉 DONE. Saved ->", OUT_PATH.resolve())