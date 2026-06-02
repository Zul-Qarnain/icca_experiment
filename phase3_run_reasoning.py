import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.preprocessing import serialize_row
from core.prompts import ZERO_SHOT, FEW_SHOT_STANDARD, FEW_SHOT_COT
from models.wrappers import call_google_studio, call_groq, call_openrouter, call_mistral, call_nvidia_nim
from phase2_run_baselines import configure_dataset, DATASETS

def main():
    os.makedirs("result", exist_ok=True)
    
    for path in DATASETS:
        if not os.path.exists(path):
            continue
            
        df, target_col, dataset_key = configure_dataset(path)
        sample = df.sample(n=100, random_state=42).reset_index(drop=True)
        results = []
        
        print(f"\n--- Running Reasoning Matrix for {dataset_key} ---")
        for i, row in sample.iterrows():
            start = time.time()
            text_row = serialize_row(row, target_col)
            true_lbl = row[target_col]
            record = {"row": i, "truth": true_lbl, "input": text_row}
            
            p_zs = ZERO_SHOT[dataset_key]
            p_fs = FEW_SHOT_STANDARD[dataset_key]
            p_cot = FEW_SHOT_COT[dataset_key]
            
            with ThreadPoolExecutor(max_workers=6) as exe:
                tasks = {
                    # Standard Models (Forced CoT via Prompt)
                    exe.submit(call_groq, p_cot, text_row, 'cot'): "Llama_CoT",
                    exe.submit(call_openrouter, p_cot, text_row, 'cot'): "Qwen_CoT",
                    exe.submit(call_google_studio, p_cot, text_row, 'cot'): "Gemma_CoT",
                    exe.submit(call_mistral, p_cot, text_row, 'cot'): "Mistral_CoT",
                    
                    # NVIDIA (Native Architectural Reasoning)
                    exe.submit(call_nvidia_nim, p_zs, text_row, True, 'standard'): "Nemotron_Native_ZS",
                    exe.submit(call_nvidia_nim, p_fs, text_row, True, 'standard'): "Nemotron_Native_FS",
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
            time.sleep(12) # Free Tier Rate Limit Protection
            
            if (i + 1) % 5 == 0:
                pd.DataFrame(results).to_csv(f"result/reasoning_{dataset_key}", index=False)

if __name__ == "__main__":
    main()
