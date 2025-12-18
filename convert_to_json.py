import pandas as pd
import json

# Read the CSV file
df = pd.read_csv('human_review_queue.csv')

# Convert to JSON format
json_data = df.to_json(orient='records', indent=2)

# Write to JSON file
with open('human_review_queue.json', 'w', encoding='utf-8') as f:
    f.write(json_data)

print(f"Converted {len(df)} records from CSV to JSON format")
print("JSON file saved as: human_review_queue.json")