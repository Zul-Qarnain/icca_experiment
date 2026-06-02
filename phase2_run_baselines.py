import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.preprocessing import serialize_row, inject_missing_data
from core.prompts import ZERO_SHOT, FEW_SHOT_STANDARD
from models.wrappers import call_google_studio, call_groq, call_openrouter, call_mistral

DATASETS = ["dataset/adult.csv", "dataset/german_credit.csv", "dataset/Heart_disease_cleveland_new.csv"]

def configure_dataset(file_path):
    df = pd.read_csv(file_path)
    base = os.path.basename(file_path)
    
    if "adult.csv" in base:
        df['income'] = df['income'].astype(str).str.strip().map({'<=50K': 0, '>50K': 1})
        df = inject_missing_data(df, 'capital-gain')
        return df, 'income', base
    elif "german" in base:
        df['class'] = df['class'].astype(str).str.strip().map({'1': 0, '2': 1})
        df = inject_missing_data(df, 'credit_amount')
        return df, 'class', base
    elif "Heart" in base:
        df['target'] = df['target'].astype(str).str.strip().map({'0': 0, '1': 1})
        df = inject_missing_data(df, 'chol')
        return df, 'target', base
    return None, None, None

def main():
    os.makedirs("result", exist_ok=True)
    
    for path in DATASETS:
        if not os.path.exists(path):
            continue
            
        df, target_col, dataset_key = configure_dataset(path)
        sample = df.sample(n=100, random_state=42).reset_index(drop=True)
        results = []
        
        print(f"\n--- Running Baselines for {dataset_key} ---")
        for i, row in sample.iterrows():
            start = time.time()
            text_row = serialize_row(row, target_col)
            true_lbl = row[target_col]
            record = {"row": i, "truth": true_lbl, "input": text_row}
            
            p_zs = ZERO_SHOT[dataset_key]
            p_fs = FEW_SHOT_STANDARD[dataset_key]
            
            with ThreadPoolExecutor(max_workers=8) as exe:
                tasks = {
                    exe.submit(call_groq, p_zs, text_row): "Llama_ZS",
                    exe.submit(call_groq, p_fs, text_row): "Llama_FS",
                    exe.submit(call_openrouter, p_zs, text_row): "Qwen_ZS",
                    exe.submit(call_openrouter, p_fs, text_row): "Qwen_FS",
                    exe.submit(call_google_studio, p_zs, text_row): "Gemma_ZS",
                    exe.submit(call_google_studio, p_fs, text_row): "Gemma_FS",
                    exe.submit(call_mistral, p_zs, text_row): "Mistral_ZS",
                    exe.submit(call_mistral, p_fs, text_row): "Mistral_FS",
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
            
            # Save row-by-row to prevent data loss
            pd.DataFrame(results).to_csv(f"result/baseline_{dataset_key}", index=False)

if __name__ == "__main__":
    main()
