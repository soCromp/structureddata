import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from handler import UnifiedDataLoader
from sklearn.preprocessing import LabelEncoder

import sys 
sys.path.append(os.path.abspath("models/CTAB-GAN-Plus"))
from model.eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics

# The complete benchmark suite
DATASETS = ['bayesian', 'cern', 'honeypot', 'lob', 'moma', 'olist', 'stroke']
MODELS = ['tabby', 'ctganp', 'icl', 'tabdiff', 'tabdlm', 'tabkg']
TRIALS = [0, 1, 2]

# The single directory containing all outputs
SYNTH_DIR = "/Users/snc/r/sd/samples" 
RESULTS_FILE = "final_evaluation_results.csv"

def evaluate_run(dataset_name, model_name, trial_num, synth_file_path):
    print(f"Evaluating {model_name} on {dataset_name} (Trial {trial_num})...")
    
    try:
        fake_df = pd.read_csv(synth_file_path)
        loader = UnifiedDataLoader(dataset_name=dataset_name, target_model_type="gan")
        real_df = loader.get_train_data()
        meta = loader.get_metadata()
        
        categorical_cols = list(meta.get('categorical', []))
        continuous_cols = list(meta.get('continuous', []))
        categorical_cols = [c for c in categorical_cols if c not in continuous_cols and c in real_df.columns]
        
        common_cols = [c for c in real_df.columns if c in fake_df.columns]
        fake_df = fake_df[common_cols]
        real_eval_df = real_df[common_cols]
        
        # 1. Run Advanced Heterogeneous Metrics FIRST
        # adv_results = evaluate_advanced_metrics(real_eval_df, fake_df, dataset_name)
        
        # --- PREPARE DATA FOR LEGACY METRICS ---
        text_cols_to_ignore = ['Title', 'request_raw']
        eval_cat_cols = [c for c in categorical_cols if c in common_cols and c not in text_cols_to_ignore]
        
        real_clean = real_eval_df.drop(columns=text_cols_to_ignore, errors='ignore').copy()
        fake_clean = fake_df.drop(columns=text_cols_to_ignore, errors='ignore').copy()
        
        # 1. STOP USING dropna() on sparse datasets! Fill NaNs safely instead.
        # Fill categorical/object columns with a string, and numerics with 0
        for col in real_clean.columns:
            if real_clean[col].dtype == 'object' or pd.api.types.is_string_dtype(real_clean[col]) or col in eval_cat_cols:
                real_clean[col] = real_clean[col].fillna("Missing")
                fake_clean[col] = fake_clean[col].fillna("Missing")
            else:
                real_clean[col] = pd.to_numeric(real_clean[col], errors='coerce').fillna(0)
                fake_clean[col] = pd.to_numeric(fake_clean[col], errors='coerce').fillna(0)
        
        # 2. Prevent bounds errors if the GAN severely collapsed
        if len(fake_clean) <= 1 or len(real_clean) <= 1:
            print(f"  [!] Skipping legacy eval: Data collapsed (Fake: {len(fake_clean)}, Real: {len(real_clean)}).")
            return {"Dataset": dataset_name, "Model": model_name, "Trial": trial_num, "Error": "Mode Collapse", **adv_results}

        # 3. Universal Label Encoding (Forces strings to integers for scaling/distances)
        for col in real_clean.columns:
            if real_clean[col].dtype == 'object' or pd.api.types.is_string_dtype(real_clean[col]) or col in eval_cat_cols:
                le = LabelEncoder()
                # Fit on combined data so unknown synthetic categories don't crash the transform
                combined = pd.concat([real_clean[col].astype(str), fake_clean[col].astype(str)], axis=0)
                le.fit(combined)
                real_clean[col] = le.transform(real_clean[col].astype(str))
                fake_clean[col] = le.transform(fake_clean[col].astype(str))
                
        target_col = meta.get('target')
        
        # 4. Shift the target column to the absolute end of the DataFrame
        if target_col and target_col in real_clean.columns:
            cols = [c for c in real_clean.columns if c != target_col] + [target_col]
            real_clean = real_clean[cols]
            fake_clean = fake_clean[cols]
            
        # 5. Save to Temporary CSVs
        temp_real = f"temp_real_{trial_num}.csv"
        temp_fake = f"temp_fake_{trial_num}.csv"
        real_clean.to_csv(temp_real, index=False)
        fake_clean.to_csv(temp_fake, index=False)
        
        # 6. Execute Legacy CTAB-GAN+ Metrics via File Paths
        try:
            s_res = stat_sim(temp_real, temp_fake, eval_cat_cols)
            stat_results = {"Avg_WD": s_res[0], "Avg_JSD": s_res[1], "Corr_Dist": s_res[2]}
        except Exception as e:
            print(f"  [!] Stat Sim failed: {e}")
            stat_results = {}
            
        try:
            if target_col and target_col in real_clean.columns:
                # Dynamic Routing: Strictly lowercase to satisfy the internal CTAB-GAN+ string matching
                prob_type_str = meta.get('type', 'classification')
                if 'regress' in prob_type_str.lower():
                    model_dict = {"regression": ["linear", "dt", "rf", "mlp"]} 
                else:
                    model_dict = {"classification": ["lr", "dt", "rf", "mlp"]}
                
                u_res = get_utility_metrics(temp_real, [temp_fake], "MinMax", model_dict, test_ratio=0.20)
                utility_results = {"Acc_or_R2": u_res[0][0], "AUC_or_RMSE": u_res[0][1], "F1_or_MAE": u_res[0][2]}
            else:
                utility_results = {"Acc_or_R2": np.nan, "AUC_or_RMSE": np.nan, "F1_or_MAE": np.nan}
        except Exception as e:
            print(f"  [!] Utility evaluation failed: {e}")
            utility_results = {}
            
        try:
            # Check for extreme mode collapse (all rows identical) which breaks NNDR
            if fake_clean.drop_duplicates().shape[0] < 2 or real_clean.drop_duplicates().shape[0] < 2:
                print("  [!] Privacy evaluation skipped: Not enough unique rows (severe mode collapse).")
                priv_results = {"DCR": np.nan, "NNDR": np.nan}
            else:
                p_res = privacy_metrics(temp_real, temp_fake)
                priv_results = {"DCR": p_res[0], "NNDR": p_res[1]}
        except Exception as e:
            print(f"  [!] Privacy evaluation failed: {e}")
            priv_results = {}

        if os.path.exists(temp_real): os.remove(temp_real)
        if os.path.exists(temp_fake): os.remove(temp_fake)

        metrics = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Trial": trial_num,
            **stat_results,
            **utility_results,
            **priv_results,
            # **adv_results
        }
        return metrics

    except Exception as e:
        print(f"  [X] FATAL ERROR evaluating {model_name} on {dataset_name} (Trial {trial_num}): {e}")
        return {"Dataset": dataset_name, "Model": model_name, "Trial": trial_num, "Error": str(e)}

if __name__ == "__main__":
    all_results = []
    
    for dataset in DATASETS:
        for model in MODELS:
            for trial in TRIALS:
                # Target format: cern.tabby.2.csv
                filename = f"{dataset}.{model}.{trial}.csv"
                expected_file = os.path.join(SYNTH_DIR, filename)
                
                if os.path.exists(expected_file):
                    metrics = evaluate_run(dataset, model, trial, expected_file)
                    all_results.append(metrics)
                else:
                    print(f"[-] Missing file: {filename}. Skipping.")
                    
    # Compile and save
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(RESULTS_FILE, index=False)
        print(f"\n✅ Evaluation complete! Master results table saved to {RESULTS_FILE}")
    else:
        print("\n[!] No synthetic data files were found. Please check the SYNTH_DIR path.")
        