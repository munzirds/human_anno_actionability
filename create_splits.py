import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load final dataset
df = pd.read_csv('final_dataset.csv')

print("=" * 60)
print("PHASE 2 — TRAIN/DEV/TEST SPLIT")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

# First split: 70% train, 30% temp (for dev+test)
X = df.drop('final_label', axis=1)
y = df['final_label']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Second split: 15% dev, 15% test (from the 30% temp)
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Combine features and labels back
train_df = pd.concat([X_train, y_train], axis=1)
dev_df = pd.concat([X_dev, y_dev], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save splits
train_df.to_csv('train_split.csv', index=False)
dev_df.to_csv('dev_split.csv', index=False)
test_df.to_csv('test_split.csv', index=False)

# Report split statistics
print(f"Total samples: {len(df):,}")
print(f"\nSplit sizes:")
print(f"  Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
print(f"  Dev:   {len(dev_df):,} ({len(dev_df)/len(df)*100:.1f}%)")
print(f"  Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")

print(f"\nLabel distribution verification:")
print("Train:")
train_dist = train_df['final_label'].value_counts().sort_index()
for label, count in train_dist.items():
    pct = count / len(train_df) * 100
    print(f"  {label}: {count:,} ({pct:.1f}%)")

print("Dev:")
dev_dist = dev_df['final_label'].value_counts().sort_index()
for label, count in dev_dist.items():
    pct = count / len(dev_df) * 100
    print(f"  {label}: {count:,} ({pct:.1f}%)")

print("Test:")
test_dist = test_df['final_label'].value_counts().sort_index()
for label, count in test_dist.items():
    pct = count / len(test_df) * 100
    print(f"  {label}: {count:,} ({pct:.1f}%)")

print(f"\nFiles created:")
print(f"  train_split.csv")
print(f"  dev_split.csv") 
print(f"  test_split.csv")

print("\n" + "=" * 60)
print("SPLIT COMPLETE - DO NOT RE-RUN")
print("=" * 60)