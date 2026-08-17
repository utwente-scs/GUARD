import json
import os
import re
import time
import requests
import pandas as pd
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv("../../.env")

CSV_PATH = "../../resources/datasets/dataset_t3.csv"
PREDICTIONS_DIR = "./predictions"
LEAF_COL = "leaf"
LABEL_COL = "label"
BATCH_SIZE = 10

MODELS = {
    "claude-sonnet-4-6": "llm_predictions_secrets_t3_claude-sonnet-4-6.csv",
    "claude-fable-5": "llm_predictions_secrets_t3_claude-fable-5.csv",
    "ollama:mistral:7b-instruct-v0.3-q4_K_M": "llm_predictions_secrets_t3_mistral.csv"
}

DEFINITION = """A hard-coded secret smell is a configuration key/parameter whose name indicates that it stores sensitive credentials directly in the IaC script: passwords, secrets, authentication tokens, API keys, private keys, SSH keys, certificate private material, or other credentials (Rahman et al., ICSE 2019)."""

def build_prompt(batch):
    numbered = "\n".join(f"{i + 1}. {leaf}" for i, leaf in enumerate(batch))
    return f"""You are classifying configuration keys extracted from Infrastructure-as-Code scripts.

{DEFINITION}

For each key below, decide whether the key name indicates a hard-coded secret (1) or not (0). Judge ONLY from the key string; you have no other context.

Keys:
{numbered}

Respond ONLY with a JSON array of exactly {len(batch)} integers (0 or 1), in the exact same order. You MUST provide exactly {len(batch)} items. Do not skip any keys. No explanation, no markdown."""

def parse_response(text, expected_n):
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON array in response: {text[:200]}")
    arr = json.loads(m.group(0))
    if len(arr) != expected_n or any(v not in (0, 1) for v in arr):
        raise ValueError(f"Malformed array (len={len(arr)})")
    return arr

def main():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    print(f"Loading dataset: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, keep_default_na=False, na_values=[])
    df[LEAF_COL] = df[LEAF_COL].astype(str)
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").fillna(-1).astype(int)
    
    labeled_df = df[df[LABEL_COL] != -1]
    unique_leaves = sorted(set(labeled_df[LEAF_COL].tolist()))
    print(f"Found {len(unique_leaves)} unique labeled leaves.")

    for model, filename in MODELS.items():
        is_ollama = model.startswith("ollama:")
        clean_model = model.replace("ollama:", "") if is_ollama else model
        
        cache_path = os.path.join(PREDICTIONS_DIR, filename)
        
        if os.path.exists(cache_path):
            cache = pd.read_csv(cache_path, dtype={LEAF_COL: str}, keep_default_na=False, na_values=[])
            done = dict(zip(cache[LEAF_COL], cache["pred"].astype(int)))
        else:
            done = {}

        todo = [leaf for leaf in unique_leaves if leaf not in done]
        print(f"\n[{model}] Leaves to classify: {len(todo)} (Cached: {len(done)})")

        if not todo:
            continue

        if not is_ollama:
            import anthropic
            client = anthropic.Anthropic()

        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start:start + BATCH_SIZE]
            prompt = build_prompt(batch)
            
            for attempt in range(3):
                try:
                    if is_ollama:
                        payload = {
                            "model": clean_model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {"temperature": 0.0}
                        }
                        resp = requests.post("http://localhost:11434/api/chat", json=payload)
                        resp.raise_for_status()
                        text = resp.json()["message"]["content"]
                    else:
                        kwargs = {
                            "model": clean_model, 
                            "max_tokens": 1000,
                            "messages": [{"role": "user", "content": prompt}]
                        }
                        if "fable" not in clean_model:
                            kwargs["temperature"] = 0
                        resp = client.messages.create(**kwargs) # type: ignore
                        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
                    
                    preds = parse_response(text, len(batch))
                    break
                except Exception as e:
                    print(f"  Retry batch {start // BATCH_SIZE}: {e}")
                    time.sleep(5 * (attempt + 1))
            else:
                print(f"Batch {start // BATCH_SIZE} failed 3 times. Skipping...")
                continue

            for leaf, p in zip(batch, preds):
                done[leaf] = int(p)
                
            pd.DataFrame({LEAF_COL: list(done.keys()), "pred": list(done.values())}).to_csv(cache_path, index=False)
            print(f"  Batch {start // BATCH_SIZE + 1}/{(len(todo) + BATCH_SIZE - 1) // BATCH_SIZE} completed")

if __name__ == "__main__":
    main()
