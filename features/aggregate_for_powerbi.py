import pandas as pd
from pathlib import Path

IN_PATH = Path("data/processed/speaker_blocks_with_sentiment.csv")
CALL_OUT = Path("data/processed/powerbi_call_level_metrics.csv")
ROLE_OUT = Path("data/processed/powerbi_role_level_metrics.csv")

def finbert_to_score(label: str) -> int:
    # Simple numeric mapping for charts + correlations
    if label == "positive":
        return 1
    if label == "negative":
        return -1
    return 0

if __name__ == "__main__":
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}. Run merge first.")

    df = pd.read_csv(IN_PATH)

    # Ensure types
    df["sentiment_vader"] = pd.to_numeric(df["sentiment_vader"], errors="coerce").fillna(0.0)
    df["finbert_confidence"] = pd.to_numeric(df["finbert_confidence"], errors="coerce").fillna(0.0)
    df["finbert_score"] = df["finbert_sentiment"].astype(str).apply(finbert_to_score)

    group_keys = ["symbol", "company_name", "year", "quarter", "date"]

    # Role-level metrics (per call + role)
    role_level = (
        df.groupby(group_keys + ["speaker_role"])
          .agg(
              blocks=("clean_text", "count"),
              avg_block_len=("block_length", "mean"),
              vader_mean=("sentiment_vader", "mean"),
              vader_median=("sentiment_vader", "median"),
              finbert_mean=("finbert_score", "mean"),
              finbert_pos=("finbert_sentiment", lambda x: (x == "positive").mean()),
              finbert_neg=("finbert_sentiment", lambda x: (x == "negative").mean()),
              finbert_neu=("finbert_sentiment", lambda x: (x == "neutral").mean()),
              finbert_avg_conf=("finbert_confidence", "mean"),
          )
          .reset_index()
    )

    ROLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    role_level.to_csv(ROLE_OUT, index=False)

    # Call-level metrics (all roles combined)
    call_level = (
        df.groupby(group_keys)
          .agg(
              total_blocks=("clean_text", "count"),
              avg_block_len=("block_length", "mean"),
              vader_mean=("sentiment_vader", "mean"),
              finbert_mean=("finbert_score", "mean"),
              finbert_pos=("finbert_sentiment", lambda x: (x == "positive").mean()),
              finbert_neg=("finbert_sentiment", lambda x: (x == "negative").mean()),
              finbert_neu=("finbert_sentiment", lambda x: (x == "neutral").mean()),
              finbert_avg_conf=("finbert_confidence", "mean"),
          )
          .reset_index()
    )

    # Management vs Analyst gap (will be weak right now because analysts are under-labeled)
    pivot = role_level.pivot_table(
        index=group_keys,
        columns="speaker_role",
        values=["vader_mean", "finbert_mean"],
        aggfunc="first"
    )

    # Flatten columns
    pivot.columns = [f"{a}__{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    # Compute gaps if both exist
    if "vader_mean__management" in pivot.columns and "vader_mean__analyst" in pivot.columns:
        pivot["vader_gap_mgmt_minus_analyst"] = pivot["vader_mean__management"] - pivot["vader_mean__analyst"]
    else:
        pivot["vader_gap_mgmt_minus_analyst"] = None

    if "finbert_mean__management" in pivot.columns and "finbert_mean__analyst" in pivot.columns:
        pivot["finbert_gap_mgmt_minus_analyst"] = pivot["finbert_mean__management"] - pivot["finbert_mean__analyst"]
    else:
        pivot["finbert_gap_mgmt_minus_analyst"] = None

    call_level = call_level.merge(
        pivot[group_keys + ["vader_gap_mgmt_minus_analyst", "finbert_gap_mgmt_minus_analyst"]],
        on=group_keys,
        how="left"
    )

    call_level.to_csv(CALL_OUT, index=False)

    print("Rows (merged speaker blocks):", df.shape)
    print("Rows (role-level):", role_level.shape)
    print("Rows (call-level):", call_level.shape)
    print("Saved ->", ROLE_OUT.resolve())
    print("Saved ->", CALL_OUT.resolve())