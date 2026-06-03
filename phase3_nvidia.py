import os
import time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.preprocessing import serialize_row
from core.prompts import ZERO_SHOT, FEW_SHOT_STANDARD
from core.parser import extract_label
from phase2_run_baselines import configure_dataset, DATASETS

# Load Key (Specified as OPENROUTER_API_KEY_PHASE3 for NVIDIA via OpenRouter)
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY_PHASE3")

def make_request_with_retry(url, headers, payload, timeout=60):
    """Exponential backoff to handle free-tier Rate Limits (HTTP 429)."""
    for attempt in range(7):
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"  [API Error] {res.status_code} - {res.text}")
                return None
        except Exception as e:
            print(f"  [Network Exception] {str(e)}")
            time.sleep(2 ** attempt)
    return None

def call_nvidia_via_openrouter(prompt_sys, prompt_user, enable_thinking=True, mode='standard'):
    # Using OpenRouter endpoint as requested
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/gemini-cli", # Required by OpenRouter
        "X-Title": "LLM Tabular Experiment"
    }
    
    # Model name on OpenRouter for Nemotron 70B
    model_name = "nvidia/llama-3.1-nemotron-70b-instruct" 
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": prompt_user}
        ],
        "temperature": 0.1
    }
    
    res = make_request_with_retry(url, headers, payload)
    if res:
        try:
            text = res["choices"][0]["message"]["content"]
            return text, extract_label(text, mode)
        except (KeyError, IndexError):
            print(f"  [Parse Error] Unexpected response format: {res}")
            return "Error", -2
    return "Error", -2

def main():
    if not OPENROUTER_KEY:
        print("Error: OPENROUTER_API_KEY_PHASE3 not found in environment.")
        return

    os.makedirs("result", exist_ok=True)
    
    for path in DATASETS:
        if not os.path.exists(path):
            continue
            
        df, target_col, dataset_key = configure_dataset(path)
        sample = df.sample(n=100, random_state=42).reset_index(drop=True)
        results = []
        
        output_file = f"result/phase3_nvidia_{dataset_key}"
        print(f"\n--- Running NVIDIA Nemotron (via OpenRouter) for {dataset_key} ---")
        
        for i, row in sample.iterrows():
            start = time.time()
            text_row = serialize_row(row, target_col)
            true_lbl = row[target_col]
            record = {"row": i, "truth": true_lbl, "input": text_row}
            
            p_zs = ZERO_SHOT[dataset_key]
            p_fs = FEW_SHOT_STANDARD[dataset_key]
            
            # NVIDIA Native Architectural Reasoning (ZS and FS)
            with ThreadPoolExecutor(max_workers=2) as exe:
                tasks = {
                    exe.submit(call_nvidia_via_openrouter, p_zs, text_row, True, 'standard'): "Nemotron_Native_ZS",
                    exe.submit(call_nvidia_via_openrouter, p_fs, text_row, True, 'standard'): "Nemotron_Native_FS",
                }
                for f in as_completed(tasks):
                    tag = tasks[f]
                    try:
                        _, lbl = f.result()
                        record[f"{tag}_label"] = lbl
                    except:
                        record[f"{tag}_label"] = -2
            
            results.append(record)
            print(f"Row {i+1}/100 completed in {time.time()-start:.2f}s")
            
            # Row-by-row checkpointing
            pd.DataFrame(results).to_csv(output_file, index=False)
            
            time.sleep(2) # Rate limit protection

if __name__ == "__main__":
    main()
