import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
INPUT_CSV = "annotated_actionability.csv"
REVIEW_CSV = "human_review_queue.csv"

CONF_THRESHOLD = 0.70      # review all low-confidence cases
A3_REVIEW_FRACTION = 0.15  # review 15% of A3 cases

RANDOM_SEED = 42

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(INPUT_CSV)

required_cols = {"usertext", "label", "confidence"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

# -----------------------------
# Initialize review flags
# -----------------------------
df["needs_human_review"] = False
df["review_reason"] = ""

# -----------------------------
# Rule 1: All ERROR rows
# -----------------------------
mask_error = df["label"] == "ERROR"
df.loc[mask_error, "needs_human_review"] = True
df.loc[mask_error, "review_reason"] = "model_error"

# -----------------------------
# Rule 2: Low-confidence cases
# -----------------------------
mask_low_conf = df["confidence"] < CONF_THRESHOLD
df.loc[mask_low_conf, "needs_human_review"] = True
df.loc[mask_low_conf, "review_reason"] += "|low_confidence"

# -----------------------------
# Rule 3: Sampled A3 cases
# -----------------------------
a3_df = df[df["label"] == "A3"]

sample_size = int(len(a3_df) * A3_REVIEW_FRACTION)

a3_sample = a3_df.sample(
    n=sample_size,
    random_state=RANDOM_SEED
)

df.loc[a3_sample.index, "needs_human_review"] = True
df.loc[a3_sample.index, "review_reason"] += "|A3_sample"

# -----------------------------
# Clean review_reason strings
# -----------------------------
df["review_reason"] = df["review_reason"].str.strip("|")

# -----------------------------
# Create human review queue
# -----------------------------
review_df = df[df["needs_human_review"]].copy()

# Add columns for annotators
review_df["human_label"] = ""
review_df["annotator_notes"] = ""

# -----------------------------
# Save outputs
# -----------------------------
review_df.to_csv(REVIEW_CSV, index=False)
df.to_csv(INPUT_CSV, index=False)

print("Human review selection complete.")
print(f"Total rows flagged for review: {len(review_df)}")
print(f"Saved review queue to: {REVIEW_CSV}")
