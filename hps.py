import os
# Must be set before importing numpy/scipy/torch to prevent MKL symbol crashes
os.environ["MKL_THREADING_LAYER"] = "GNU"

import optuna
# from optuna.storages import RDBStorage
from optuna.storages import JournalStorage, JournalFileStorage, JournalFileSymlinkLock
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import sys
import math
import shutil

from handler import UnifiedDataLoader

SEARCH_SPACES = {
    'tabby': {
        "lr": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        "model": ['l1']
    },
    'ctganp': {
        "batch_size": [128, 256, 512]
    },
    'icl': {
        "k_shots": [5],
        "temperature": [0.2, 0.6, 1.0]
    },
    'tabdiff': {    
        "lr": [1e-5, 1e-4, 1e-3],
    },
    'tabdlm': {
        "batch_accum": [4, 8, 16],
    },
    'tabkg': {
        "temp_range": [
            "0.1,0.2,0.3,0.4,0.5",
            "0.3,0.4,0.5,0.6,0.7",
            "0.6,0.7,0.8,0.9,1.0"
        ]
    }
}

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
    cwd = None
    space = SEARCH_SPACES[model_type] # Grab the dictionary for this model
    
    if model_type == 'tabby':
        lr = trial.suggest_categorical("lr", space["lr"])
        model = trial.suggest_categorical("model", space["model"])
        max_epochs = 10
        if dataset == 'cern':
            max_epochs = 2
        elif dataset == 'olist':
            max_epochs = 4
        elif dataset == 'bayesian':
            max_epochs = 7
        
        checkpoint_dir = f"/staging/c/cromp/sd/tabby/{dataset}/optuna_{trial.number}"
        
        traincmd = [
            "python", "trainplain.py",
            "-t", 
            "-p", checkpoint_dir,
            "-d", dataset,
            "-e", str(max_epochs),
            "-lr", str(lr),
            f"-{model}", 
            "--local", "-eff", "-mh",
            '-n', '200',
        ]
        
        cmds = [traincmd]
        cwd = 'models/tabby'
        synth_path = f"{checkpoint_dir}/samplesclean.csv"
        
    elif model_type == 'ctganp':
        batch_size = trial.suggest_categorical("batch_size", space["batch_size"])

        cmd = ["python", "run_ctganp.py", dataset, "--bs", str(batch_size), "--id", str(trial.number)]
        cmds = [cmd]
        cwd = 'models/CTAB-GAN-Plus'
        synth_path = f"synth/{dataset}/ctganp_0.csv"

    elif model_type == 'icl':
        k_shots = trial.suggest_categorical("k_shots", space["k_shots"])
        temperature = trial.suggest_categorical("temperature", space["temperature"])
        
        cmd = [
            "python", "models/ICL/run_icl.py",
            "--dataset", dataset,
            "--num_samples", "200",
            "--k_shots", str(k_shots),
            "--temperature", str(temperature),
            "--model_id", "/staging/groups/cs_geodes/zoo/Meta-Llama-3-8B"
        ]
        cmds = [cmd]
        synth_path = f"synth/{dataset}/icl.csv"
        
    elif model_type == 'tabdiff':
        lr = trial.suggest_categorical("lr", space["lr"])
        
        # lr gets specified in a config file we need to make. copy over from template
        source_file = "models/TabDiff/tabdiff/configs/hps_tabdiff_configs.toml"
        dest_file = "models/TabDiff/tabdiff/configs/tabdiff_configs.toml"
        shutil.copy2(source_file, dest_file)
        with open(dest_file, "a") as file: # append the lr 
            file.write(f"lr = {str(lr)}\n")
        
        cmdtrain = [
            sys.executable, "main.py",
            "--dataname", dataset,
            "--mode", "train",
            "--exp_name", f"hps_{str(trial.number)}",
        ]
        cmdtest = [
            sys.executable, "main.py",
            "--dataname", dataset,
            "--mode", "test", "--no_wandb",
            "--exp_name", f"hps_{str(trial.number)}",
            "--ckpt_path", f'./tabdiff/ckpt/{dataset}/hps_{str(trial.number)}/model_8000.pt'
        ]
        cmds = [cmdtrain, cmdtest]
        cwd = 'models/TabDiff'
        synth_path = f"models/TabDiff/synthetic/{dataset}/test.csv"

    elif model_type == 'tabdlm':
        batch_accum = trial.suggest_categorical("batch_accum", space["batch_accum"])
        
        if dataset in ['stroke', 'cern', 'olist', 'bayesian']:
            batch_size = 32
        elif dataset == 'lob':
            batch_size = 16
        elif dataset in ['moma', 'honeypot']:
            batch_size = 2
        
        cmdtrain = [
            "python", "main.py", 
            "train",
            "--dataset_name", dataset,
            "--description", f"_tabdlm_{trial.number}",
            "--epochs", "15",
            "--batch_size", str(batch_size),
            "--batch_accum", str(batch_size*batch_accum),
            "--lora_r", "4",
            "--lora_alpha", "128",
            "--answer_len", "90",
            "--loss_type", "no_divide_pmask",
            "--bf16"
        ]
        cmdsample = [
            "python", "main.py",
            "sample",
            "--dataset_name", dataset,
            "--description", f"_tabdlm_{trial.number}",
            "--save_description", f"_tabdlm_synth_{trial.number}",
            "--do_sampling", "--bf16",
            "--gen_length", "90", "--block_length", "90",
            "--sample_step", "90", "--temperature", "1.0",
            "--sample_batch_size", "32", "--seed", str(0),
            "--n", "200"
        ]
    #     TOKENIZERS_PARALLELISM=false PYTHONPATH=. python main.py train --dataset_name $dataset \
    #             --description "_tabdlm" --epochs 1 --batch_size 2 --batch_accum 32 \
    #             --loss_type no_divide_pmask  --lora_r 16 --lora_alpha 64 --bf16
    # TOKENIZERS_PARALLELISM=false PYTHONPATH=. python main.py sample --dataset_name $dataset \
    #             --description "_tabdlm" --save_description "_tabdlm_synth" --do_sampling \
    #             --temperature 1.0 --sample_batch_size 32 --seed 1 --n 32
        
        cmds = [cmdtrain, cmdsample]
        cwd = 'models/TabDLM'
        synth_path = f"models/TabDLM/result/{dataset}/synthetic_result/_tabdlm_{trial.number}_tabdlm_synth_{trial.number}.csv"

    elif model_type == 'tabkg':
        temp_range = trial.suggest_categorical("temp_range", space["temp_range"])
        
        cmd = [
            "python", "main.py",
            "--data", dataset,
            "--method", "crkg",
            "--ensemble", "gpt,gpt,gpt,gpt,gpt",
            "--temp_range", temp_range,
            '--id', str(trial.number)
        ]

        cmds = [cmd]
        cwd = 'models/TabKG'
        synth_path = f"models/TabKG/results/{dataset}/CRKG_FilteredOutput_{trial.number}.csv"
        
    print(f"\n--- Starting Trial {trial.number} for {model_type} on {dataset} ---")
    
    for cmd in cmds:
        subprocess.run(cmd, cwd=cwd, check=True, capture_output=False,
            env=os.environ.copy())
    
    try:
        synth_df = pd.read_csv(synth_path)
    except FileNotFoundError:
        return 1000
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
        # os.makedirs(drive, exist_ok=True)
    
    # Create the Optuna study
    study_name = f"hpo_{target_model}_{target_dataset}"
    study_path = os.path.join(drive, f'{study_name}.log')
    
    # Force the NFS-safe Symlink lock
    lock = JournalFileSymlinkLock(filepath=study_path)
    
    # Wrap in Optuna's Journal backend
    storage = JournalStorage(JournalFileStorage(study_path, lock_obj=lock))
    
    search_space = SEARCH_SPACES.get(target_model, {})
    total_trials = math.prod([len(values) for values in search_space.values()]) if search_space else 1
    
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize", # We want the lowest KS distance + lowest penalty
        storage=storage,
        sampler=sampler,
        load_if_exists=True
    )
    
    study.optimize(lambda t: objective(t, target_dataset, target_model), n_trials=total_trials)
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        print("\n=== Best Hyperparameters Found ===")
        print(study.best_params)
        print(f"Best Score: {study.best_value}")
    else:
        print("\n[WARNING] No trials completed successfully. Check logs for errors.")
    