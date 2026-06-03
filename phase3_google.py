import os
import time
import pandas as pd
import requests
from core.preprocessing import serialize_row
from core.prompts import FEW_SHOT_COT
from core.parser import extract_label
from phase2_run_baselines import configure_dataset, DATASETS

# Load Key
GEMINI_KEY = os.getenv("GOOGLE_API_KEY_PHASE3")

def make_request_with_retry(url, headers, payload, timeout=30):
    """Exponential backoff to handle free-tier Rate Limits (HTTP 429)."""
    for attempt in range(7):
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 429:
                time.sleep(2 ** attempt) # Sleep 1, 2, 4, 8, 16 seconds
                continue
            else:
                print(f"  [API Error] {res.status_code} - {res.text}")
                return None
        except Exception as e:
            print(f"  [Network Exception] {str(e)}")
            time.sleep(2 ** attempt)
    return None

def call_google_studio(prompt_sys, prompt_user, mode='standard'):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_KEY}"
    payload = {
        "contents": [{"parts": [{"text": f"{prompt_sys}\n\nInput: {prompt_user}"}]}],
        "generationConfig": {"temperature": 0.1}
    }
    res = make_request_with_retry(url, {"Content-Type": "application/json"}, payload)
    if res:
        try:
            text = res["candidates"][0]["content"]["parts"][0]["text"]
            return text, extract_label(text, mode)
        except (KeyError, IndexError):
            print(f"  [Parse Error] Unexpected response format: {res}")
            return "Error", -2
    return "Error", -2

def main():
    if not GEMINI_KEY:
        print("Error: GOOGLE_API_KEY_PHASE3 not found in environment.")
        return

    os.makedirs("result", exist_ok=True)
    
    for path in DATASETS:
        if not os.path.exists(path):
            continue
            
        df, target_col, dataset_key = configure_dataset(path)
        sample = df.sample(n=100, random_state=42).reset_index(drop=True)
        results = []
        
        output_file = f"result/phase3_google_{dataset_key}"
        print(f"\n--- Running Google Gemini Reasoning for {dataset_key} ---")
        
        for i, row in sample.iterrows():
            start = time.time()
            text_row = serialize_row(row, target_col)
            true_lbl = row[target_col]
            record = {"row": i, "truth": true_lbl, "input": text_row}
            
            p_cot = FEW_SHOT_COT[dataset_key]
            
            # Single model call
            _, lbl = call_google_studio(p_cot, text_row, 'cot')
            record["Gemma_CoT_label"] = lbl
            
            results.append(record)
            print(f"Row {i+1}/100 completed in {time.time()-start:.2f}s")
            
            # Row-by-row checkpointing
            pd.DataFrame(results).to_csv(output_file, index=False)
            
            time.sleep(1) # Minimal delay for Google

if __name__ == "__main__":
    main()
