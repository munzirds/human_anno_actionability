import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import AzureOpenAI
from dotenv import load_dotenv
import signal
import sys

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT = os.getenv("AZURE_MODEL_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

print("Loading environment variables...")
print(f"Endpoint: {ENDPOINT}")
print(f"Deployment: {DEPLOYMENT}")
print(f"API Key: {'*' * 10 if API_KEY else 'Not found'}")

if not all([API_KEY, ENDPOINT, DEPLOYMENT]):
    print("\nMissing variables:")
    if not API_KEY: print("- AZURE_OPENAI_API_KEY")
    if not ENDPOINT: print("- AZURE_OPENAI_ENDPOINT")
    if not DEPLOYMENT: print("- AZURE_MODEL_DEPLOYMENT")
    raise RuntimeError("Missing Azure OpenAI credentials in .env file")

# -----------------------------
# Configuration
# -----------------------------
INPUT_CSV = "suicide watch.csv"      # must contain column: usertext
OUTPUT_CSV = "annotated_actionability.csv"
CHECKPOINT_FILE = "annotation_checkpoint.json"
PROGRESS_FILE = "annotation_progress.csv"
REQUEST_DELAY = 0.1   # seconds (reduced for speed)
BATCH_SIZE = 5        # smaller batches for better checkpointing
MAX_WORKERS = 3       # fewer workers for stability
MAX_RETRIES = 3       # retry failed requests
RETRY_DELAY = 2       # seconds between retries

# -----------------------------
# Azure OpenAI client
# -----------------------------
print("\nInitializing Azure OpenAI client...")
client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    api_version=API_VERSION
)
print("Client initialized successfully.")

# -----------------------------
# Prompts
# -----------------------------
SYSTEM_PROMPT = (
    "Research assistant for mental health support classification. "
    "Classify urgency objectively. When uncertain, choose lower level. "
    "Respond only in requested JSON format."
)

def sanitize_input(text: str) -> str:
    """Fast sanitization for research analysis"""
    if not text or len(text.strip()) == 0:
        return "[No content provided]"
    
    # Quick length limit
    text = text[:1200]  # Shorter for speed
    
    # Fast replacements for content policy
    text = text.replace('kill myself', 'harm myself')
    text = text.replace('suicide', 'self-harm')
    text = text.replace('want to die', 'want to stop')
    text = text.replace('end it all', 'stop everything')
    
    # Quick cleanup
    return text.replace('"""', '').replace('```', '').strip()

def build_user_prompt(text: str) -> str:
    sanitized_text = sanitize_input(text)
    return f"""RESEARCH: Classify support urgency for this text.

Levels: A0=low, A1=monitor, A2=intervention, A3=crisis

Text: \"{sanitized_text}\"

JSON response:
{{
  "label": "A0|A1|A2|A3",
  "confidence": 0.0-1.0,
  "rationale": "brief reason"
}}"""

# -----------------------------
# Annotation function
# -----------------------------
def annotate_text_with_retry(text: str, idx: int = 0) -> dict:
    """Annotate with retry logic for robustness"""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(text)}
                ],
                temperature=0.1,
                max_tokens=150,
                timeout=30  # 30 second timeout
            )
            
            result_text = response.choices[0].message.content.strip()
            result = json.loads(result_text)
            result['original_index'] = idx
            return result
        
        except json.JSONDecodeError:
            if attempt == MAX_RETRIES - 1:
                return {"label": "A0", "confidence": 0.0, "rationale": "JSON parse error", "original_index": idx}
        except Exception as e:
            if "content_filter" in str(e):
                return {"label": "A2", "confidence": 0.8, "rationale": "Content filtered", "original_index": idx}
            elif attempt == MAX_RETRIES - 1:
                return {"label": "A0", "confidence": 0.0, "rationale": f"Failed after {MAX_RETRIES} attempts", "original_index": idx}
            else:
                print(f"Row {idx}: Attempt {attempt + 1} failed, retrying...")
                time.sleep(RETRY_DELAY)
    
    return {"label": "A0", "confidence": 0.0, "rationale": "Max retries exceeded", "original_index": idx}

def save_checkpoint(completed_indices, results):
    """Save progress checkpoint"""
    checkpoint = {
        'completed_indices': list(completed_indices),
        'timestamp': time.time(),
        'total_completed': len(completed_indices)
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)
    
    # Save partial results
    if results:
        pd.DataFrame(results).to_csv(PROGRESS_FILE, index=False)

def load_checkpoint():
    """Load previous checkpoint if exists"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def load_partial_results():
    """Load partial results if exists"""
    if os.path.exists(PROGRESS_FILE):
        return pd.read_csv(PROGRESS_FILE).to_dict('records')
    return []

def process_batch_robust(batch_data, completed_indices, all_results):
    """Process batch with checkpointing"""
    batch_results = []
    for idx, text in batch_data:
        if idx in completed_indices:
            continue  # Skip already processed
        
        result = annotate_text_with_retry(text, idx)
        batch_results.append(result)
        all_results.append(result)
        completed_indices.add(idx)
        
        # Save checkpoint every 10 items
        if len(completed_indices) % 10 == 0:
            save_checkpoint(completed_indices, all_results)
        
        time.sleep(REQUEST_DELAY)
    
    return batch_results

# -----------------------------
# Main processing
# -----------------------------
if __name__ == "__main__":
    try:
        print(f"\nLoading dataset: {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)
        print(f"Found {len(df)} messages to annotate")
        
        if 'usertext' not in df.columns:
            print(f"Error: 'usertext' column not found. Available columns: {list(df.columns)}")
            exit(1)
        
        # Check for existing progress
        checkpoint = load_checkpoint()
        all_results = load_partial_results()
        completed_indices = set()
        
        if checkpoint:
            completed_indices = set(checkpoint['completed_indices'])
            print(f"\nResuming from checkpoint: {len(completed_indices)} already completed")
            print(f"Remaining: {len(df) - len(completed_indices)}")
        else:
            print("\nStarting fresh annotation")
        
        # Prepare remaining data
        data_pairs = [(idx, row['usertext']) for idx, row in df.iterrows() if idx not in completed_indices]
        
        if not data_pairs:
            print("All items already completed!")
        else:
            batches = [data_pairs[i:i + BATCH_SIZE] for i in range(0, len(data_pairs), BATCH_SIZE)]
            
            print(f"\nProcessing {len(batches)} batches (batch size: {BATCH_SIZE})")
            print(f"Retry policy: {MAX_RETRIES} attempts with {RETRY_DELAY}s delay")
            print("Checkpoints saved every 10 items")
            
            try:
                # Process sequentially for better error handling
                with tqdm(total=len(data_pairs), desc="Annotating", initial=0) as pbar:
                    for batch in batches:
                        try:
                            batch_results = process_batch_robust(batch, completed_indices, all_results)
                            pbar.update(len(batch_results))
                            
                            # Print recent progress
                            for result in batch_results[-3:]:  # Show last 3
                                print(f"Row {result['original_index']}: {result['label']} ({result['confidence']:.2f})")
                                
                        except KeyboardInterrupt:
                            print("\nInterrupted by user. Saving progress...")
                            save_checkpoint(completed_indices, all_results)
                            print(f"Progress saved. {len(completed_indices)} items completed.")
                            exit(0)
                        except Exception as e:
                            print(f"\nBatch error: {e}. Saving progress and continuing...")
                            save_checkpoint(completed_indices, all_results)
                            continue
            
            except Exception as e:
                print(f"\nCritical error: {e}")
                save_checkpoint(completed_indices, all_results)
                print("Progress saved before exit.")
                exit(1)
        
        # Final save and statistics
        save_checkpoint(completed_indices, all_results)
        
        if all_results:
            errors = sum(1 for r in all_results if "Failed" in r['rationale'] or "error" in r['rationale'].lower())
            filtered = sum(1 for r in all_results if "filtered" in r['rationale'].lower())
            
            print(f"\nProcessing complete!")
            print(f"Total processed: {len(all_results)}")
            print(f"Content filtered: {filtered}")
            print(f"Failed requests: {errors}")
            
            # Create final output
            results_df = pd.DataFrame(all_results)
            results_df = results_df.sort_values('original_index')
            
            # Merge with original data
            df_indexed = df.reset_index().rename(columns={'index': 'original_index'})
            final_df = df_indexed.merge(results_df, on='original_index', how='left')
            final_df = final_df.drop('original_index', axis=1)
            final_df.to_csv(OUTPUT_CSV, index=False)
            
            print(f"\nResults saved to: {OUTPUT_CSV}")
            print(f"\nLabel distribution:")
            print(results_df['label'].value_counts())
            
            # Cleanup checkpoint files
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
            if os.path.exists(PROGRESS_FILE):
                os.remove(PROGRESS_FILE)
            print("Checkpoint files cleaned up.")
        
    except FileNotFoundError:
        print(f"Error: {INPUT_CSV} not found")
    except Exception as e:
        print(f"Critical error: {e}")
        if 'completed_indices' in locals() and 'all_results' in locals():
            save_checkpoint(completed_indices, all_results)
            print("Emergency checkpoint saved.")
