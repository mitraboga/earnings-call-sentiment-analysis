from datasets import load_dataset
import pandas as pd
from pathlib import Path

DATASET_NAME = "kurry/sp500_earnings_transcripts"

if __name__ == "__main__":
    out_path = Path("data/raw/transcripts_raw.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("CWD:", Path.cwd())
    print("Loading dataset:", DATASET_NAME)

    ds = load_dataset(DATASET_NAME)
    print("Dataset splits:", list(ds.keys()))

    df = pd.DataFrame(ds["train"])
    print("Loaded shape:", df.shape)
    print("Columns:", df.columns.tolist())

    df.to_csv(out_path, index=False)
    print("Saved file exists?", out_path.exists())
    print("Saved ->", out_path.resolve())
