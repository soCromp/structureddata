import os
# Must be set before importing numpy/scipy/torch to prevent MKL symbol crashes
os.environ["MKL_THREADING_LAYER"] = "GNU"

import optuna
from optuna.storages import RDBStorage
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import sys

# Import your existing loader to get the real data
from handler import UnifiedDataLoader

def calculate_fast_proxy(real_df, synth_df, dataset_name):
    """
    The composite metric: KS Distance + (Penalty * Causal Violations)
    Optuna will try to minimize this score.
    """
    # 1. Marginal Distribution Match (KS Test)
    ks_scores = []
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns
    
    # We only KS test columns that exist in the synthetic output to prevent crashing
    # on broken generative trials.
    valid_numeric_cols = [c for c in numeric_cols if c in synth_df.columns]
    
    for col in valid_numeric_cols:
        # Drop NaNs safely before testing
        real_vals = real_df[col].dropna()
        synth_vals = pd.to_numeric(synth_df[col], errors='coerce').dropna()
        
        if len(synth_vals) > 0 and len(real_vals) > 0:
            stat, _ = ks_2samp(real_vals, synth_vals)
            ks_scores.append(stat)
        else:
            ks_scores.append(1.0) # Maximum penalty if a numeric column completely collapses
            
    avg_ks_distance = np.mean(ks_scores) if ks_scores else 1.0

    # 2. Causal Invariant Penalty
    penalty = 0.0
    
    try:
        if dataset_name == 'olist':
            # TRAP: Delivery must happen after purchase
            if 'order_delivered_customer_date' in synth_df.columns and 'order_purchase_timestamp' in synth_df.columns:
                # In your handler, 0.0 was used to fill NaT. Ignore zeros.
                valid_mask = (synth_df['order_delivered_customer_date'] > 0) & (synth_df['order_purchase_timestamp'] > 0)
                if valid_mask.any():
                    violations = synth_df.loc[valid_mask, 'order_delivered_customer_date'] < synth_df.loc[valid_mask, 'order_purchase_timestamp']
                    penalty = violations.mean() * 10.0
                
        elif dataset_name == 'bayesian':
            # TRAP: X3 must be X1 + X2 (+ random noise)
            if all(c in synth_df.columns for c in ['collider', 'root_1', 'root_2']):
                expected_x3 = synth_df['root_1'] + synth_df['root_2']
                mse_drift = ((synth_df['collider'] - expected_x3)**2).mean()
                penalty = mse_drift * 2.0
                
        elif dataset_name == 'cern':
            # TRAP: Non-Linear Trigonometric Physics
            # Invariant Mass (M) = sqrt(2 * pt1 * pt2 * (cosh(eta1 - eta2) - cos(phi1 - phi2)))
            req_cols = ['M_', 'pt1_', 'pt2_', 'eta1_', 'eta2_', 'phi1_', 'phi2_']
            if all(c in synth_df.columns for c in req_cols):
                # Force numeric in case a model generated strings
                s = synth_df[req_cols].apply(pd.to_numeric, errors='coerce').dropna()
                
                # The term inside the sqrt must not be negative (models might hallucinate physics that result in imaginary numbers)
                inner_term = 2 * s['pt1_'] * s['pt2_'] * (np.cosh(s['eta1_'] - s['eta2_']) - np.cos(s['phi1_'] - s['phi2_']))
                
                # Heavy penalty for imaginary mass
                imaginary_violations = (inner_term < 0).mean()
                
                # Drift penalty for valid masses
                valid_inner = inner_term[inner_term >= 0]
                valid_M = s.loc[inner_term >= 0, 'M_']
                expected_M = np.sqrt(valid_inner)
                
                # Use Mean Absolute Error (MAE) for drift so massive outliers don't cause float overflow
                mass_drift = np.abs(valid_M - expected_M).mean() if len(valid_M) > 0 else 100.0
                
                # Combine imaginary violations and kinematic drift
                penalty = (imaginary_violations * 20.0) + (mass_drift / 10.0) 

        elif dataset_name == 'lob':
            # TRAP: The Deterministic Math Trap
            # Distances and Notionals (Volume) cannot be negative
            dist_notional_cols = [c for c in synth_df.columns if 'distance' in c or 'notional' in c]
            if dist_notional_cols:
                s = synth_df[dist_notional_cols].apply(pd.to_numeric, errors='coerce')
                # Count percentage of generated numbers that are physically impossible (negative)
                violations = (s < 0).sum().sum() / (s.shape[0] * s.shape[1])
                penalty = violations * 10.0
                
        elif dataset_name == 'stroke':
            # TRAP: Medical Imbalance and Biological Bounds
            if 'stroke' in synth_df.columns:
                # Mode Collapse Penalty: Stroke rate should be ~5%. 
                # If a GAN pushes this to 50/50, it gets penalized heavily.
                stroke_rate = pd.to_numeric(synth_df['stroke'], errors='coerce').mean()
                target_rate = 0.05
                collapse_penalty = np.abs(stroke_rate - target_rate) * 10.0
                
                # Biological Bound Penalty: BMI cannot be negative
                bmi_violations = 0
                if 'bmi' in synth_df.columns:
                    bmi_vals = pd.to_numeric(synth_df['bmi'], errors='coerce')
                    bmi_violations = (bmi_vals < 0).mean() * 5.0
                    
                penalty = collapse_penalty + bmi_violations
                
        elif dataset_name == 'honeypot':
            # TRAP: Extreme Sparsity & Nested Schema
            # If the payload features (request_raw) exist, network features (attackerPort) are generally missing, and vice versa.
            if 'request_raw' in synth_df.columns and 'attackerPort' in synth_df.columns:
                has_payload = (synth_df['request_raw'] != "Missing") & synth_df['request_raw'].notna()
                has_network = (synth_df['attackerPort'] != "Missing") & synth_df['attackerPort'].notna()
                
                # Models that fail to learn conditional sparsity will populate both columns simultaneously
                mutual_violations = (has_payload & has_network).mean()
                penalty = mutual_violations * 10.0
                
        elif dataset_name == 'moma':
            # TRAP: High-Cardinality Text Overwhelming
            # Since MoMA lacks strict mathematical invariants, we test format/missingness collapse.
            # If the model starts hallucinating pure NaNs/Empty Strings for the complex text descriptions:
            if 'Title' in synth_df.columns:
                empty_titles = synth_df['Title'].replace(["Missing", "", "NaN", "None"], np.nan).isna().mean()
                penalty = empty_titles * 5.0
                
    except Exception as e:
        # If the synthetic data is so corrupted that the math checks crash, 
        # issue a massive penalty so Optuna discards the trial.
        print(f"Warning: Penalty calculation failed due to corrupted generated data. Error: {e}")
        penalty = 50.0 

    # Combine for the final objective score
    total_score = avg_ks_distance + penalty
    print(f"Trial Score: {total_score:.4f} (KS: {avg_ks_distance:.4f}, Penalty: {penalty:.4f})")
    
    if np.isnan(total_score):
        return 100.0
        
    return total_score


def objective(trial, dataset, model_type):
    """
    The core Optuna loop. It samples HPs, runs your existing code, and scores it.
    """
    # --- 1. Sample Hyperparameters based on Model Type ---
    if model_type == 'tabby':
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        epochs = trial.suggest_int("epochs", 1, 5)
        
        checkpoint_dir = f"/mnt/data/sonia/sd/tabby/{dataset}/optuna_{trial.number}"
        
        # 1. Train Command
        train_cmd = [
            "python", "models/tabby/trainplain.py",
            "-t", 
            "-p", checkpoint_dir,
            "-d", dataset,
            "-e", str(epochs),
            "-l1", str(lr),
            "--local", "-eff"
        ]
        
        # 2. Sample Command (Assuming this reads from the checkpoint and outputs samplesclean.csv)
        sample_cmd = [
            "python", "models/tabby/trainplain.py",
            "-p", checkpoint_dir,
            "-d", dataset,
            "-n", "50", # or however many you need for the micro-sample
            "--local", "-l1", "-eff"
        ]
        
        cmds = [train_cmd, sample_cmd]
        
        # # Run sequentially
        # subprocess.run(train_cmd, check=True)
        # subprocess.run(sample_cmd, check=True)
        
        # Read directly from the checkpoint directory, not the synth directory
        synth_path = f"{checkpoint_dir}/samplesclean.csv"
        
        
    elif model_type == 'ctganp':
        batch_size = trial.suggest_categorical("batch_size", [500, 1000, 2000])
        # You would need to temporarily modify run_ctganp.py to accept batch_size as an arg
        cmd = ["python", "models/CTAB-GAN-Plus/run_ctganp.py", "1", dataset, "--bs", str(batch_size)]
        cmds = [cmd]
        synth_path = f"synth/{dataset}/ctganp_optuna_{trial.number}.csv"
        
        
    elif model_type == 'icl':
        # Lock k_shots but sweep three distinct entropy levels
        k_shots = trial.suggest_categorical("k_shots", [5])
        temperature = trial.suggest_categorical("temperature", [0.2, 0.6, 1.0])
        
        cmd = [
            "python", "models/ICL/run_icl.py",
            "--dataset", dataset,
            "--num_samples", "50",
            "--k_shots", str(k_shots),
            "--temperature", str(temperature),
            "--model_id", "/mnt/data/zoo/meta-llama/Meta-Llama-3-8B"
        ]
        cmds = [cmd]
        
        synth_path = f"synth/{dataset}/icl.csv"
        

    # --- 2. Execute the Training Script ---
    print(f"\n--- Starting Trial {trial.number} for {model_type} on {dataset} ---")
    
    try:
        for cmd in cmds:
            # Run the bash command. This blocks until the model finishes training and sampling
            subprocess.run(cmd, check=True, capture_output=False)
        
        # NOTE: You must ensure your training script saves the micro-sample to `synth_path`
        synth_df = pd.read_csv(synth_path)
        
    except subprocess.CalledProcessError as e:
        # If the model OOMs or crashes, tell Optuna to abandon this hyperparameter branch
        raise optuna.TrialPruned()
    except FileNotFoundError:
        print("Model failed to output CSV.")
        raise optuna.TrialPruned()

    # --- 3. Evaluate the Fast Proxy ---
    # Load the real dataset from your existing handler cache
    loader = UnifiedDataLoader(dataset_name=dataset, target_model_type="llm")
    real_df = loader.get_train_data()
    
    score = calculate_fast_proxy(real_df, synth_df, dataset)
    
    return score


if __name__ == "__main__":
    target_dataset = sys.argv[1] # e.g., 'olist'
    target_model = sys.argv[2].lower()   # e.g., 'tabby'
    if len(sys.argv) == 3:
        drive = ''
    else:
        drive = sys.argv[3]
        os.makedirs(drive, exist_ok=True)
    
    # Create the Optuna study
    study_name = f"hpo_{target_model}_{target_dataset}"
    study_path = os.path.join(drive, study_name)
    
    # We use SQLite so if the cluster crashes, you don't lose your HP progress!
    storage_url = f"sqlite:///{study_path}.db"
    storage = RDBStorage(
        url=storage_url,
        engine_kwargs={"connect_args": {"timeout": 60.0}} # prevent lock issues
    )
    
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize", # We want the lowest KS distance + lowest penalty
        storage=storage,
        load_if_exists=True
    )
    
    # Run 5 trials (1 for ICL)
    if target_model == 'icl':
        study.optimize(lambda t: objective(t, target_dataset, target_model), n_trials=1)
    else:
        study.optimize(lambda t: objective(t, target_dataset, target_model), n_trials=5)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        print("\n=== Best Hyperparameters Found ===")
        print(study.best_params)
        print(f"Best Score: {study.best_value}")
    else:
        print("\n[WARNING] No trials completed successfully. Check logs for errors.")
    