import os
import time
import pandas as pd
import requests
from core.preprocessing import serialize_row
from core.prompts import FEW_SHOT_COT
from core.parser import extract_label
from phase2_run_baselines import configure_dataset, DATASETS

# Load Key
GROQ_KEY = os.getenv("GROQ_API_KEY_PHASE3")

def make_request_with_retry(url, headers, payload, timeout=30):
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

def call_groq(prompt_sys, prompt_user, mode='standard'):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "system", "content": prompt_sys}, {"role": "user", "content": prompt_user}],
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
    if not GROQ_KEY:
        print("Error: GROQ_API_KEY_PHASE3 not found in environment.")
        return

    os.makedirs("result", exist_ok=True)
    
    for path in DATASETS:
        if not os.path.exists(path):
            continue
            
        df, target_col, dataset_key = configure_dataset(path)
        sample = df.sample(n=100, random_state=42).reset_index(drop=True)
        results = []
        
        output_file = f"result/phase3_groq_{dataset_key}"
        print(f"\n--- Running Groq Reasoning for {dataset_key} ---")
        
        for i, row in sample.iterrows():
            start = time.time()
            text_row = serialize_row(row, target_col)
            true_lbl = row[target_col]
            record = {"row": i, "truth": true_lbl, "input": text_row}
            
            p_cot = FEW_SHOT_COT[dataset_key]
            
            # Single model call
            _, lbl = call_groq(p_cot, text_row, 'cot')
            record["Llama_CoT_label"] = lbl
            
            results.append(record)
            print(f"Row {i+1}/100 completed in {time.time()-start:.2f}s")
            
            # Row-by-row checkpointing
            pd.DataFrame(results).to_csv(output_file, index=False)
            
            time.sleep(10) # Free Tier Rate Limit Protection

if __name__ == "__main__":
    main()
