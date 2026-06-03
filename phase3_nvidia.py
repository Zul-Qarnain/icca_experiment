import os
import time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.preprocessing import serialize_row
from core.prompts import ZERO_SHOT, FEW_SHOT_STANDARD
from core.parser import extract_label
from phase2_run_baselines import configure_dataset, DATASETS

# Load Key (Official Enterprise NVIDIA API Key)
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")

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

def call_nvidia_nim(prompt_sys, prompt_user, mode='standard'):
    # Official NVIDIA NIM Enterprise API (OpenAI-compatible)
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {NVIDIA_KEY}",
        "Content-Type": "application/json"
    }
    
    # Updated model for Enterprise NIM
    model_name = "nvidia/nemotron-3-super-120b-a12b" 
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": prompt_user}
        ],
        "temperature": 0.1,
        "max_tokens": 1024
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
    if not NVIDIA_KEY:
        print("Error: NVIDIA_API_KEY not found in environment.")
        return

    os.makedirs("result", exist_ok=True)
    
    for path in DATASETS:
        if not os.path.exists(path):
            continue
            
        df, target_col, dataset_key = configure_dataset(path)
        sample = df.sample(n=100, random_state=42).reset_index(drop=True)
        results = []
        
        output_file = f"result/phase3_nvidia_{dataset_key}"
        print(f"\n--- Running NVIDIA Enterprise NIM for {dataset_key} ---")
        
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
                    exe.submit(call_nvidia_nim, p_zs, text_row, 'standard'): "Nemotron_Native_ZS",
                    exe.submit(call_nvidia_nim, p_fs, text_row, 'standard'): "Nemotron_Native_FS",
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
            
            time.sleep(1) # Enterprise API generally has higher limits

if __name__ == "__main__":
    main()
