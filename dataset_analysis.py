import pandas as pd
import numpy as np
from collections import Counter
import re

# Load final dataset
df = pd.read_csv('final_dataset.csv')

print("=" * 60)
print("PHASE 1 — DATASET CHARACTERIZATION")
print("=" * 60)

# 1.1 Dataset Statistics
print("\n1.1 DATASET STATISTICS")
print("-" * 30)

# Total samples
total_samples = len(df)
print(f"Total samples: {total_samples:,}")

# Label distribution
print(f"\nLabel distribution:")
label_counts = df['final_label'].value_counts().sort_index()
for label, count in label_counts.items():
    pct = (count / total_samples) * 100
    print(f"  {label}: {count:,} ({pct:.1f}%)")

# Average text length (tokens - approximate using whitespace)
df['token_count'] = df['usertext'].str.split().str.len()
avg_tokens = df['token_count'].mean()
median_tokens = df['token_count'].median()
print(f"\nText length statistics:")
print(f"  Average tokens: {avg_tokens:.1f}")
print(f"  Median tokens: {median_tokens:.1f}")
print(f"  Min tokens: {df['token_count'].min()}")
print(f"  Max tokens: {df['token_count'].max()}")

# Human review coverage
human_reviewed = df['human_label'].notna().sum()
human_pct = (human_reviewed / total_samples) * 100
print(f"\nHuman review coverage:")
print(f"  Human reviewed: {human_reviewed:,} ({human_pct:.1f}%)")
print(f"  Model only: {total_samples - human_reviewed:,} ({100 - human_pct:.1f}%)")

# Disagreements (LLM vs human)
disagreements = df[df['human_label'].notna() & (df['label'] != df['human_label'])]
disagreement_count = len(disagreements)
disagreement_pct = (disagreement_count / human_reviewed) * 100 if human_reviewed > 0 else 0
print(f"\nDisagreements (LLM vs Human):")
print(f"  Disagreements: {disagreement_count} / {human_reviewed} ({disagreement_pct:.1f}%)")
print(f"  Agreement: {human_reviewed - disagreement_count} / {human_reviewed} ({100 - disagreement_pct:.1f}%)")

# 1.2 Agreement Analysis
print("\n1.2 AGREEMENT ANALYSIS")
print("-" * 30)

if human_reviewed > 0:
    # Overall agreement
    agreements = df[df['human_label'].notna() & (df['label'] == df['human_label'])]
    overall_agreement = len(agreements) / human_reviewed * 100
    print(f"Overall agreement: {overall_agreement:.1f}%")
    
    # Agreement on A3 (critical cases)
    a3_cases = df[df['human_label'].notna() & ((df['label'] == 'A3') | (df['human_label'] == 'A3'))]
    a3_agreements = df[df['human_label'].notna() & (df['label'] == 'A3') & (df['human_label'] == 'A3')]
    if len(a3_cases) > 0:
        a3_agreement_pct = len(a3_agreements) / len(a3_cases) * 100
        print(f"Agreement on A3 cases: {len(a3_agreements)} / {len(a3_cases)} ({a3_agreement_pct:.1f}%)")
    
    # Disagreement patterns
    if disagreement_count > 0:
        print(f"\nDisagreement patterns:")
        disagreement_patterns = disagreements.groupby(['label', 'human_label']).size().sort_values(ascending=False)
        for (llm_label, human_label), count in disagreement_patterns.items():
            pct = (count / disagreement_count) * 100
            print(f"  {llm_label} -> {human_label}: {count} ({pct:.1f}%)")
        
        # Focus on A1↔A2 disagreements
        a1_a2_disagreements = disagreements[
            ((disagreements['label'] == 'A1') & (disagreements['human_label'] == 'A2')) |
            ((disagreements['label'] == 'A2') & (disagreements['human_label'] == 'A1'))
        ]
        if len(a1_a2_disagreements) > 0:
            a1_a2_pct = (len(a1_a2_disagreements) / disagreement_count) * 100
            print(f"\nA1↔A2 disagreements: {len(a1_a2_disagreements)} / {disagreement_count} ({a1_a2_pct:.1f}% of all disagreements)")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)