import json
import pandas as pd
import sys

# -----------------------------
# Configuration
# -----------------------------
ANNOTATED_CSV = "annotated_actionability.csv"  # Base annotated dataset
REVIEWED_JSON = "reviewed_output.json"  # Human corrections
OUTPUT_CSV = "final_dataset.csv"

REQUIRED_COLUMNS = {
    "usertext",
    "label"
}

VALID_LABELS = {"A0", "A1", "A2", "A3"}

# -----------------------------
# Load annotated dataset and human corrections
# -----------------------------
try:
    # Load base annotated dataset
    df = pd.read_csv(ANNOTATED_CSV)
    print(f"Loaded {len(df)} records from annotated dataset")
    
    # Load human corrections if they exist
    try:
        with open(REVIEWED_JSON, "r", encoding="utf-8") as f:
            reviewed_data = json.load(f)
        reviewed_df = pd.DataFrame(reviewed_data)
        print(f"Loaded {len(reviewed_df)} human corrections")
    except FileNotFoundError:
        print(f"[WARNING] No human corrections found at {REVIEWED_JSON}")
        reviewed_df = pd.DataFrame()
        
except Exception as e:
    print(f"[ERROR] Failed to load datasets: {e}")
    sys.exit(1)

# -----------------------------
# Validate structure
# -----------------------------
missing_cols = REQUIRED_COLUMNS - set(df.columns)
if missing_cols:
    print(f"[ERROR] Missing required columns: {missing_cols}")
    sys.exit(1)

# -----------------------------
# Merge human reviews with original dataset
# -----------------------------
if not reviewed_df.empty and 'usertext' in reviewed_df.columns:
    # Create a mapping of usertext to human_label from reviews
    human_labels = reviewed_df[reviewed_df['human_label'].notna() & (reviewed_df['human_label'] != '')]
    human_label_map = dict(zip(human_labels['usertext'], human_labels['human_label']))
    
    # Add human_label column to original dataset
    df['human_label'] = df['usertext'].map(human_label_map)
    print(f"Merged {len(human_label_map)} human labels into original dataset")
else:
    df['human_label'] = None
    print("No human labels to merge")

# -----------------------------
# Create final_label
# -----------------------------
df["final_label"] = df["human_label"].where(
    df["human_label"].notna() & (df["human_label"] != ""),
    df["label"]
)

# -----------------------------
# Sanity checks
# -----------------------------
invalid_labels = set(df["final_label"]) - VALID_LABELS
if invalid_labels:
    print(f"[ERROR] Invalid labels found in final_label: {invalid_labels}")
    sys.exit(1)

# Check missing text
empty_text = df["usertext"].isna().sum() + (df["usertext"].str.strip() == "").sum()
if empty_text > 0:
    print(f"[WARNING] {empty_text} rows have empty usertext")

# -----------------------------
# Save frozen dataset
# -----------------------------
df.to_csv(OUTPUT_CSV, index=False)

# -----------------------------
# Summary (for logs / paper)
# -----------------------------
print("Label freezing complete.")
print(f"Base dataset: {ANNOTATED_CSV}")
print(f"Human corrections: {REVIEWED_JSON}")
print(f"Output file: {OUTPUT_CSV}")
print("\nFinal label distribution:")
print(df["final_label"].value_counts().sort_index())

print("\nHuman review coverage:")
human_reviewed = df['human_label'].notna().sum()
print(f"Human reviewed: {human_reviewed} / {len(df)} ({human_reviewed/len(df)*100:.1f}%)")
print(f"Model only: {len(df) - human_reviewed} / {len(df)} ({(len(df) - human_reviewed)/len(df)*100:.1f}%)")

print("\nDataset is now FROZEN.")
print("Do NOT modify `final_label` after this point.")
