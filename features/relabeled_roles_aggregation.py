import pandas as pd
import re
from pathlib import Path

IN_PATH = Path("data/processed/speaker_blocks_with_sentiment.csv")

ROLE_OUT = Path("data/processed/powerbi_role_level_metrics_v2.csv")
CALL_OUT = Path("data/processed/powerbi_call_level_metrics_v2.csv")

# --- Stronger heuristics for finance earnings calls ---
BANK_FIRMS = [
    "goldman", "morgan", "j.p.", "jp morgan", "barclays", "citi", "citigroup",
    "ubs", "wells fargo", "bofa", "bank of america", "deutsche", "credit suisse",
    "jefferies", "raymond james", "piper sandler", "canaccord", "rbc", "td",
    "scotiabank", "bmo", "jpm", "hsbc", "societe generale", "sg", "evercore",
    "baird", "stifel", "keybanc", "cowen", "guggenheim"
]

MGMT_TITLES = [
    "ceo", "cfo", "coo", "cto", "chief", "president", "chairman", "chair",
    "vice president", "vp", "svp", "evp", "head of", "general manager", "gm"
]

OPERATOR_WORDS = ["operator", "moderator", "coordinator", "conference call"]

QA_CUES = [
    "question", "my first question", "my next question", "two questions",
    "can you talk about", "can you discuss", "could you elaborate", "what is the outlook",
    "thank you for taking", "i have a question"
]

def finbert_to_score(label: str) -> int:
    if label == "positive":
        return 1
    if label == "negative":
        return -1
    return 0

def looks_like_analyst(speaker: str, text: str) -> bool:
    s = (speaker or "").lower()
    t = (text or "").lower()

    # Bank/firm name in speaker line (very common)
    if any(f in s for f in BANK_FIRMS):
        return True

    # Analyst-y cues in the text
    if any(cue in t for cue in QA_CUES) and len(t.split()) > 20:
        return True

    # Typical pattern: "Name - Firm"
    if re.search(r"\s-\s", speaker or "") and any(f in s for f in BANK_FIRMS):
        return True

    return False

def looks_like_management(speaker: str) -> bool:
    s = (speaker or "").lower()
    return any(k in s for k in MGMT_TITLES)

def looks_like_operator(speaker: str) -> bool:
    s = (speaker or "").lower()
    return any(k in s for k in OPERATOR_WORDS)

def detect_section(text: str) -> str:
    """
    Coarse section split:
    - Q&A if it looks like questions
    - otherwise Prepared Remarks
    """
    t = (text or "").lower()
    if any(cue in t for cue in QA_CUES):
        return "Q&A"
    return "Prepared Remarks"

if __name__ == "__main__":
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}. Run merge first.")

    df = pd.read_csv(IN_PATH)

    # Ensure columns exist
    for c in ["speaker", "clean_text", "speaker_role", "sentiment_vader", "finbert_sentiment", "finbert_confidence"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # Normalize
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["sentiment_vader"] = pd.to_numeric(df["sentiment_vader"], errors="coerce").fillna(0.0)
    df["finbert_confidence"] = pd.to_numeric(df["finbert_confidence"], errors="coerce").fillna(0.0)
    df["finbert_score"] = df["finbert_sentiment"].astype(str).apply(finbert_to_score)

    # --- relabel roles ---
    new_roles = []
    for sp, tx in zip(df["speaker"].astype(str), df["clean_text"]):
        if looks_like_operator(sp):
            new_roles.append("operator")
        elif looks_like_management(sp):
            new_roles.append("management")
        elif looks_like_analyst(sp, tx):
            new_roles.append("analyst")
        else:
            new_roles.append("other")

    df["speaker_role_v2"] = new_roles

    # --- detect section ---
    df["section"] = df["clean_text"].apply(detect_section)

    group_keys = ["symbol", "company_name", "year", "quarter", "date"]

    # ROLE-LEVEL with section split
    role_level = (
        df.groupby(group_keys + ["speaker_role_v2", "section"])
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

    # CALL-LEVEL overall
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

    # Management vs Analyst gap (v2)
    pivot = role_level.pivot_table(
        index=group_keys,
        columns="speaker_role_v2",
        values=["vader_mean", "finbert_mean"],
        aggfunc="mean"
    )

    pivot.columns = [f"{a}__{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    if "vader_mean__management" in pivot.columns and "vader_mean__analyst" in pivot.columns:
        pivot["vader_gap_mgmt_minus_analyst_v2"] = pivot["vader_mean__management"] - pivot["vader_mean__analyst"]
    else:
        pivot["vader_gap_mgmt_minus_analyst_v2"] = None

    if "finbert_mean__management" in pivot.columns and "finbert_mean__analyst" in pivot.columns:
        pivot["finbert_gap_mgmt_minus_analyst_v2"] = pivot["finbert_mean__management"] - pivot["finbert_mean__analyst"]
    else:
        pivot["finbert_gap_mgmt_minus_analyst_v2"] = None

    call_level = call_level.merge(
        pivot[group_keys + ["vader_gap_mgmt_minus_analyst_v2", "finbert_gap_mgmt_minus_analyst_v2"]],
        on=group_keys,
        how="left"
    )

    ROLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    role_level.to_csv(ROLE_OUT, index=False)
    call_level.to_csv(CALL_OUT, index=False)

    print("✅ Relabeled roles distribution (v2):")
    print(df["speaker_role_v2"].value_counts().to_string())
    print("\n✅ Section distribution:")
    print(df["section"].value_counts().to_string())

    print("\nSaved ->", ROLE_OUT.resolve())
    print("Saved ->", CALL_OUT.resolve())