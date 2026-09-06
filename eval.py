import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

from handler import UnifiedDataLoader

import dython.nominal
if not hasattr(dython.nominal, 'compute_associations'):
    def compute_associations_shim(*args, **kwargs):
        kwargs['compute_only'] = True
        return dython.nominal.associations(*args, **kwargs)['corr']
    dython.nominal.compute_associations = compute_associations_shim
sys.path.append(os.path.abspath("models/CTAB-GAN-Plus"))
from model.eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics

from multimodal_realism import evaluate_multimodal_realism
from mle import evaluate_mle
from manual import evaluate_domain_constraints

# The complete benchmark suite
DATASETS = ['honeypot', 'stroke', 'cern', 'lob', 'moma', 'olist', 'bayesian']
MODELS = ['tabby', 'ctganp', 'icl', 'tabdiff', 'tabdlm', 'tabkg']
TRIALS = [1, 2, 3]

# The single directory containing all outputs
SYNTH_DIR = "/home/sonia/samples"
RESULTS_FILE = "final_evaluation_results.csv"

# Known freeform/unstructured text fields across datasets
KNOWN_TEXT_COLS = ['Title', 'request_raw']


def evaluate_run(dataset_name, model_name, trial_num, synth_file_path):
    print(f"Evaluating {model_name} on {dataset_name} (Trial {trial_num})...")

    try:
        fake_df = pd.read_csv(synth_file_path)
        if len(fake_df) > 2000:
            fake_df = fake_df.iloc[-2000:]
        loader = UnifiedDataLoader(dataset_name=dataset_name, target_model_type="llm")
        real_df = loader.get_train_data()
        meta = loader.get_metadata()

        categorical_cols = list(meta.get('categorical', []))
        continuous_cols = list(meta.get('continuous', []))
        categorical_cols = [c for c in categorical_cols if c not in continuous_cols and c in real_df.columns]

        # ---------------------------------------------------------------------
        # 1. EVALUATE MULTIMODAL REALISM (CROSS-MODAL JOINT FIDELITY)
        # ---------------------------------------------------------------------
        multimodal_results = {}
        try:
            real_mm = real_df.copy()
            fake_mm = fake_df.copy()

            expected_text_cols = [c for c in KNOWN_TEXT_COLS if c in real_mm.columns]

            for text_col in expected_text_cols:
                if text_col not in fake_mm.columns:
                    fake_mm[text_col] = ""
                else:
                    fake_mm[text_col] = fake_mm[text_col].fillna("")

            all_target_cols = [c for c in real_mm.columns if c in fake_mm.columns]
            real_mm = real_mm[all_target_cols]
            fake_mm = fake_mm[all_target_cols]

            multimodal_results = evaluate_multimodal_realism(
                real_df=real_mm,
                synth_df=fake_mm,
                categorical_cols=categorical_cols,
                continuous_cols=continuous_cols
            )
        except Exception as e:
            print(f"  [!] Multimodal realism evaluation failed: {e}")
            multimodal_results = {
                "Joint_FID": np.nan,
                "C2ST_Accuracy": np.nan,
                "C2ST_AUC": np.nan
            }

        # ---------------------------------------------------------------------
        # EVALUATE DOWNSTREAM ML UTILITY (TSTR)
        # ---------------------------------------------------------------------
        real_test_df = loader.get_test_data()
        mle_results = {}
        try:
            mle_results = evaluate_mle(
                synth_train=fake_df, 
                real_test=real_test_df, 
                real_train_reference=real_df, 
                meta=meta
            )
        except Exception as e:
            print(f"  [!] MLE TSTR evaluation failed: {e}")
            
        # ---------------------------------------------------------------------
        # DOMAIN CONSTRAINTS
        # ---------------------------------------------------------------------
        try:
            domain_results = evaluate_domain_constraints(fake_df, dataset_name)
        except Exception as e:
            print(f"  [!] Tier 4 Domain Constraint evaluation failed: {e}")
            domain_results = {"Constraint_Violation_Rate": np.nan}

        # ---------------------------------------------------------------------
        # COMPILE ALL METRICS
        # ---------------------------------------------------------------------
        metrics = {
            "Dataset": dataset_name,
            "Model": model_name,
            "Trial": trial_num,
            **multimodal_results,
            **mle_results,
            **domain_results,
        }
        return metrics

    except Exception as e:
        print(f"  [X] FATAL ERROR evaluating {model_name} on {dataset_name} (Trial {trial_num}): {e}")
        return {
            "Dataset": dataset_name,
            "Model": model_name,
            "Trial": trial_num,
            "Error": str(e)
        }


if __name__ == "__main__":
    all_results = []

    for dataset in DATASETS:
        for model in MODELS:
            for trial in TRIALS:
                if model == 'tabkg' and trial > 1:
                    continue
                
                filename = f"{dataset}.{model}.{trial}.csv"
                expected_file = os.path.join(SYNTH_DIR, filename)

                if os.path.exists(expected_file):
                    metrics = evaluate_run(dataset, model, trial, expected_file)
                    all_results.append(metrics)
                else:
                    print(f"[-] Missing file: {filename}. Skipping.")

    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        
        # Enforce list order for processing
        results_df['Dataset'] = pd.Categorical(results_df['Dataset'], categories=DATASETS, ordered=True)
        results_df['Model'] = pd.Categorical(results_df['Model'], categories=MODELS, ordered=True)
        
        # 1. Save Raw CSV
        results_df.to_csv(RESULTS_FILE, index=False)
        print(f"\n✅ Raw master results table saved to {RESULTS_FILE}")
        
        numeric_cols = [c for c in results_df.columns if c not in ['Dataset', 'Model', 'Trial', 'Error']]
        for col in numeric_cols:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
            
        # Extract means and standard deviations
        means = results_df.groupby(['Dataset', 'Model'], observed=True)[numeric_cols].mean()
        stds = results_df.groupby(['Dataset', 'Model'], observed=True)[numeric_cols].std().fillna(0.0)

        # 2. Generate Human Readable Text
        try:
            formatted_df = pd.DataFrame(index=means.index)
            
            def format_text_num(val):
                if pd.isnull(val): return "NaN"
                if abs(val) >= 1e6: return f"{val:.2e}"
                return f"{val:.2f}"
            
            for col in numeric_cols:
                m_series = means[col]
                s_series = stds[col]
                formatted_df[col] = [f"{format_text_num(m)} ± {format_text_num(s)}" if pd.notnull(m) else "NaN" for m, s in zip(m_series, s_series)]
                
            txt_file = "final_evaluation_summary.txt"
            with open(txt_file, "w") as f:
                f.write("=== HETEROGENEOUS GENERATIVE MODELS BENCHMARK ===\n")
                f.write("Aggregated over trials (Mean ± Std)\n")
                f.write("=" * 120 + "\n\n")
                f.write(formatted_df.reset_index().to_string(index=False, justify='left', col_space=10))
            print(f"📄 Formatted summary saved to {txt_file}")
        except Exception as e:
            print(f"  [!] Failed to generate summary text file: {e}")

        # 3. Generate LaTeX Tables for Paper
        try:
            print("Generating LaTeX tables...")
            latex_file = "evaluation_tables.tex"
            
            higher_is_better = ['MLE_Accuracy', 'MLE_F1', 'MLE_AUC', 'MLE_R2']
            closer_to_half = ['C2ST_Accuracy', 'C2ST_AUC']
            
            # Enforce strict capitalization for the paper
            MODEL_DISPLAY_NAMES = {
                'tabby': 'Tabby',
                'ctganp': 'CTGANP',
                'icl': 'ICL',
                'tabdiff': 'TabDiff',
                'tabdlm': 'TabDLM',
                'tabkg': 'TabKG'
            }

            def format_latex_cell(m, s, is_best):
                if pd.isnull(m):
                    return "-"
                
                # Handle large numbers with grouped scientific notation
                max_val = max(abs(m), abs(s)) if pd.notnull(s) else abs(m)
                if max_val >= 1e6:
                    exp = int(np.floor(np.log10(max_val)))
                    m_base = m / (10**exp)
                    s_base = s / (10**exp)
                    # If standard deviation is 0, format it cleanly
                    if s == 0:
                        inner_str = f"{m_base:.2f} \\times 10^{{{exp}}} \\pm 0.00"
                    else:
                        inner_str = f"({m_base:.2f} \\pm {s_base:.2f}) \\times 10^{{{exp}}}"
                else:
                    inner_str = f"{m:.2f} \\pm {s:.2f}"
                    
                if is_best:
                    return f"$\\bf{{{inner_str}}}$"
                return f"${inner_str}$"

            # 1. Define specific dataset subsets for the specialized tables
            # Maintain the original DATASETS order
            class_datasets = [d for d in DATASETS if d in ['honeypot', 'stroke', 'moma', 'bayesian']]
            reg_datasets = [d for d in DATASETS if d in ['cern', 'lob', 'olist']]
            constraint_datasets = [d for d in DATASETS if d not in ['moma', 'stroke']]

            # 2. Helper function to generate any LaTeX table dynamically
            def write_latex_table(f, metric_name, dataset_list, caption, label):
                f.write(f"% ==========================================\n")
                f.write(f"% Table for {metric_name}\n")
                f.write(f"% ==========================================\n")
                f.write("\\begin{table}[h]\n\\centering\n")
                
                col_format = "l" + "c" * len(dataset_list)
                f.write(f"\\begin{{tabular}}{{{col_format}}}\n\\toprule\n")
                
                header = ["Model"] + [d.capitalize() for d in dataset_list]
                f.write(" & ".join(header) + " \\\\\n\\midrule\n")
                
                # Determine best model (excluding TabKG to prevent rewarding memorization)
                best_model_per_dataset = {}
                for dataset in dataset_list:
                    if dataset in means.index.get_level_values(0):
                        dataset_means = means.loc[dataset, metric_name].dropna()
                        
                        if 'tabkg' in dataset_means.index:
                            dataset_means = dataset_means.drop('tabkg')
                            
                        if not dataset_means.empty:
                            if metric_name in higher_is_better:
                                best_model_per_dataset[dataset] = dataset_means.idxmax()
                            elif metric_name in closer_to_half:
                                best_model_per_dataset[dataset] = (dataset_means - 0.5).abs().idxmin()
                            else:
                                best_model_per_dataset[dataset] = dataset_means.idxmin()
                
                # Write rows
                for model in MODELS:
                    display_name = MODEL_DISPLAY_NAMES.get(model, model)
                    row_data = [display_name]
                    for dataset in dataset_list:
                        try:
                            m = means.loc[(dataset, model), metric_name]
                            s = stds.loc[(dataset, model), metric_name]
                            is_best = best_model_per_dataset.get(dataset) == model
                            row_data.append(format_latex_cell(m, s, is_best))
                        except KeyError:
                            row_data.append("-")
                            
                    f.write(" & ".join(row_data) + " \\\\\n")
                    
                f.write("\\bottomrule\n\\end{tabular}\n")
                f.write(f"\\caption{{{caption}}}\n")
                f.write(f"\\label{{{label}}}\n")
                f.write("\\end{table}\n\n\n")

            # 3. Generate all tables
            with open(latex_file, "w") as f:
                
                # Write specialized tables
                write_latex_table(f, 'MLE_Accuracy', class_datasets, 
                                  'Downstream ML Efficacy (Accuracy) for classification tasks.', 'tab:mle_class')
                
                write_latex_table(f, 'MLE_RMSE', reg_datasets, 
                                  'Downstream ML Efficacy (RMSE) for regression tasks.', 'tab:mle_reg')
                
                write_latex_table(f, 'Constraint_Violation_Rate', constraint_datasets, 
                                  'Domain Constraint Violation Rate (Lower is better).', 'tab:constraints')

                # Write standard tables for all remaining metrics across all datasets
                skip_metrics = ['MLE_Accuracy', 'MLE_RMSE', 'Constraint_Violation_Rate']
                
                for metric in numeric_cols:
                    if metric in skip_metrics:
                        continue
                    
                    clean_metric = metric.replace('_', '\\_')
                    caption = f"Benchmark results for {clean_metric} (Mean $\\pm$ Std)."
                    label = f"tab:{metric.lower()}"
                    
                    write_latex_table(f, metric, DATASETS, caption, label)
                    
            print(f"📑 LaTeX tables successfully generated and saved to {latex_file}")
            
        except Exception as e:
            print(f"  [!] Failed to generate LaTeX file: {e}")

    else:
        print("\n[!] No synthetic data files were found. Please check the SYNTH_DIR path.")