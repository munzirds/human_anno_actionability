import pandas as pd
import numpy as np

# Load annotated data
try:
    df = pd.read_csv("annotated_actionability.csv")
    print(f"Loaded {len(df)} annotated items")
except FileNotFoundError:
    print("No annotated data found. Please run annotation first.")
    exit(1)

# Select items for human review based on criteria:
# 1. Low confidence predictions (< 0.7)
# 2. Random sample of high-confidence items
# 3. All A3 (crisis) classifications
# 4. Content-filtered items

review_queue = []

# Low confidence items
low_conf = df[df['confidence'] < 0.7]
review_queue.append(low_conf)
print(f"Added {len(low_conf)} low-confidence items")

# All A3 items (critical)
a3_items = df[df['label'] == 'A3']
review_queue.append(a3_items)
print(f"Added {len(a3_items)} A3 (crisis) items")

# Content filtered items
filtered_items = df[df['rationale'].str.contains('filtered', case=False, na=False)]
review_queue.append(filtered_items)
print(f"Added {len(filtered_items)} content-filtered items")

# Random sample of remaining high-confidence items (10%)
remaining = df[~df.index.isin(pd.concat(review_queue).index)]
sample_size = max(10, int(len(remaining) * 0.1))
random_sample = remaining.sample(n=min(sample_size, len(remaining)), random_state=42)
review_queue.append(random_sample)
print(f"Added {len(random_sample)} random sample items")

# Combine and deduplicate
final_queue = pd.concat(review_queue).drop_duplicates()
final_queue = final_queue.reset_index(drop=True)

# Add review reason
final_queue['review_reason'] = ''
final_queue.loc[final_queue['confidence'] < 0.7, 'review_reason'] = 'Low confidence'
final_queue.loc[final_queue['label'] == 'A3', 'review_reason'] = 'Crisis level'
final_queue.loc[final_queue['rationale'].str.contains('filtered', case=False, na=False), 'review_reason'] = 'Content filtered'
final_queue.loc[final_queue['review_reason'] == '', 'review_reason'] = 'Random sample'

# Save review queue
final_queue.to_csv("human_review_queue.csv", index=False)
print(f"\nCreated review queue with {len(final_queue)} items")
print(f"Saved to: human_review_queue.csv")

# Show distribution
print(f"\nReview reasons:")
print(final_queue['review_reason'].value_counts())
print(f"\nLabel distribution:")
print(final_queue['label'].value_counts())