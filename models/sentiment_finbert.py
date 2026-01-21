import pandas as pd
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

IN_PATH = Path("data/processed/speaker_blocks_cleaned.csv")
OUT_PATH = Path("data/processed/speaker_blocks_with_finbert.csv")
CHECKPOINT_PATH = Path("data/processed/finbert_checkpoint.txt")

MODEL_NAME = "ProsusAI/finbert"
LABELS = ["negative", "neutral", "positive"]

CHUNK_ROWS = 500         # read CSV in chunks
BATCH_SIZE = 16          # CPU-safe
MAX_LEN = 256            # keep smaller for speed

def finbert_predict_batch(texts, tokenizer, model, device):
    # Tokenize
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    conf, idx = torch.max(probs, dim=1)
    labels = [LABELS[i] for i in idx.cpu().numpy().tolist()]
    confs = conf.cpu().numpy().tolist()
    return labels, confs

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

    if start_row > 0 and not OUT_PATH.exists():
        print("⚠️ Checkpoint exists but output file missing. Restarting from 0.")
        start_row = 0

    if start_row == 0 and OUT_PATH.exists():
        print("⚠️ Output file already exists and checkpoint is 0.")
        print("   Delete data/processed/speaker_blocks_with_finbert.csv if you want a clean rebuild.")
        print("   OR delete data/processed/finbert_checkpoint.txt to force rebuild.")
        raise SystemExit("Stopping to prevent duplicate append. Clean the output/checkpoint and rerun.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print(f"▶️ FinBERT starting from row {start_row}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    total_processed = start_row
    first_write = not OUT_PATH.exists()

    reader = pd.read_csv(IN_PATH, chunksize=CHUNK_ROWS)

    # Skip already processed rows (chunk skipping)
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
            chunk = chunk.iloc[rows_to_skip:].copy()
            rows_to_skip = 0

            # Process this partial chunk
            texts = chunk["clean_text"].fillna("").astype(str).tolist()

            labels_out = []
            confs_out = []

            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]

                # If blank text, force neutral with low confidence
                cleaned_batch = [b for b in batch if b.strip()]
                if not cleaned_batch:
                    labels_out.extend(["neutral"] * len(batch))
                    confs_out.extend([0.0] * len(batch))
                    continue

                # Predict only on non-empty, then map back
                pred_labels, pred_confs = finbert_predict_batch(cleaned_batch, tokenizer, model, device)

                # Map predictions back to original batch shape
                it = iter(zip(pred_labels, pred_confs))
                for b in batch:
                    if b.strip():
                        lab, cf = next(it)
                        labels_out.append(lab)
                        confs_out.append(cf)
                    else:
                        labels_out.append("neutral")
                        confs_out.append(0.0)

            chunk["finbert_sentiment"] = labels_out
            chunk["finbert_confidence"] = confs_out

            chunk.to_csv(OUT_PATH, mode="a", header=first_write, index=False)
            first_write = False
            total_processed += len(chunk)

            CHECKPOINT_PATH.write_text(str(total_processed))
            print(f"✅ FinBERT processed rows: {total_processed}")

    # Process remaining chunks
    try:
        for chunk in reader:
            chunk = chunk.copy()
            texts = chunk["clean_text"].fillna("").astype(str).tolist()

            labels_out = []
            confs_out = []

            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]

                cleaned_batch = [b for b in batch if b.strip()]
                if not cleaned_batch:
                    labels_out.extend(["neutral"] * len(batch))
                    confs_out.extend([0.0] * len(batch))
                    continue

                pred_labels, pred_confs = finbert_predict_batch(cleaned_batch, tokenizer, model, device)

                it = iter(zip(pred_labels, pred_confs))
                for b in batch:
                    if b.strip():
                        lab, cf = next(it)
                        labels_out.append(lab)
                        confs_out.append(cf)
                    else:
                        labels_out.append("neutral")
                        confs_out.append(0.0)

            chunk["finbert_sentiment"] = labels_out
            chunk["finbert_confidence"] = confs_out

            chunk.to_csv(OUT_PATH, mode="a", header=first_write, index=False)
            first_write = False
            total_processed += len(chunk)

            CHECKPOINT_PATH.write_text(str(total_processed))
            print(f"✅ FinBERT processed rows: {total_processed}")

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user (CTRL+C).")
        print(f"✅ Progress saved. Next run will resume from row {total_processed}.")
        sys.exit(0)

    print("🎉 DONE. Saved ->", OUT_PATH.resolve())