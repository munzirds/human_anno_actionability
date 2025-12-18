import pandas as pd
import numpy as np
from collections import Counter
import re

# Load final dataset
df = pd.read_csv('final_dataset.csv')

# Prepare output
output_lines = []

output_lines.append("=" * 60)
output_lines.append("PHASE 1 — DATASET CHARACTERIZATION")
output_lines.append("=" * 60)

# 1.1 Dataset Statistics
output_lines.append("\n1.1 DATASET STATISTICS")
output_lines.append("-" * 30)

# Total samples
total_samples = len(df)
output_lines.append(f"Total samples: {total_samples:,}")

# Label distribution
output_lines.append(f"\nLabel distribution:")
label_counts = df['final_label'].value_counts().sort_index()
for label, count in label_counts.items():
    pct = (count / total_samples) * 100
    output_lines.append(f"  {label}: {count:,} ({pct:.1f}%)")

# Average text length (tokens - approximate using whitespace)
df['token_count'] = df['usertext'].str.split().str.len()
avg_tokens = df['token_count'].mean()
median_tokens = df['token_count'].median()
output_lines.append(f"\nText length statistics:")
output_lines.append(f"  Average tokens: {avg_tokens:.1f}")
output_lines.append(f"  Median tokens: {median_tokens:.1f}")
output_lines.append(f"  Min tokens: {df['token_count'].min()}")
output_lines.append(f"  Max tokens: {df['token_count'].max()}")

# Human review coverage
human_reviewed = df['human_label'].notna().sum()
human_pct = (human_reviewed / total_samples) * 100
output_lines.append(f"\nHuman review coverage:")
output_lines.append(f"  Human reviewed: {human_reviewed:,} ({human_pct:.1f}%)")
output_lines.append(f"  Model only: {total_samples - human_reviewed:,} ({100 - human_pct:.1f}%)")

# Disagreements (LLM vs human)
disagreements = df[df['human_label'].notna() & (df['label'] != df['human_label'])]
disagreement_count = len(disagreements)
disagreement_pct = (disagreement_count / human_reviewed) * 100 if human_reviewed > 0 else 0
output_lines.append(f"\nDisagreements (LLM vs Human):")
output_lines.append(f"  Disagreements: {disagreement_count} / {human_reviewed} ({disagreement_pct:.1f}%)")
output_lines.append(f"  Agreement: {human_reviewed - disagreement_count} / {human_reviewed} ({100 - disagreement_pct:.1f}%)")

# 1.2 Agreement Analysis
output_lines.append("\n1.2 AGREEMENT ANALYSIS")
output_lines.append("-" * 30)

if human_reviewed > 0:
    # Overall agreement
    agreements = df[df['human_label'].notna() & (df['label'] == df['human_label'])]
    overall_agreement = len(agreements) / human_reviewed * 100
    output_lines.append(f"Overall agreement: {overall_agreement:.1f}%")
    
    # Agreement on A3 (critical cases)
    a3_cases = df[df['human_label'].notna() & ((df['label'] == 'A3') | (df['human_label'] == 'A3'))]
    a3_agreements = df[df['human_label'].notna() & (df['label'] == 'A3') & (df['human_label'] == 'A3')]
    if len(a3_cases) > 0:
        a3_agreement_pct = len(a3_agreements) / len(a3_cases) * 100
        output_lines.append(f"Agreement on A3 cases: {len(a3_agreements)} / {len(a3_cases)} ({a3_agreement_pct:.1f}%)")
    
    # Disagreement patterns
    if disagreement_count > 0:
        output_lines.append(f"\nDisagreement patterns:")
        disagreement_patterns = disagreements.groupby(['label', 'human_label']).size().sort_values(ascending=False)
        for (llm_label, human_label), count in disagreement_patterns.items():
            pct = (count / disagreement_count) * 100
            output_lines.append(f"  {llm_label} -> {human_label}: {count} ({pct:.1f}%)")
        
        # Focus on A1↔A2 disagreements
        a1_a2_disagreements = disagreements[
            ((disagreements['label'] == 'A1') & (disagreements['human_label'] == 'A2')) |
            ((disagreements['label'] == 'A2') & (disagreements['human_label'] == 'A1'))
        ]
        if len(a1_a2_disagreements) > 0:
            a1_a2_pct = (len(a1_a2_disagreements) / disagreement_count) * 100
            output_lines.append(f"\nA1<->A2 disagreements: {len(a1_a2_disagreements)} / {disagreement_count} ({a1_a2_pct:.1f}% of all disagreements)")

output_lines.append("\n" + "=" * 60)
output_lines.append("ANALYSIS COMPLETE")
output_lines.append("=" * 60)

# Print to console
for line in output_lines:
    print(line)

# Save to file
with open('dataset_analysis_results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print("\nResults saved to: dataset_analysis_results.txt")