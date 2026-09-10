import os
import numpy as np
import pandas as pd

# The complete benchmark suite
DATASETS = ['honeypot', 'stroke', 'cern', 'lob', 'moma', 'olist', 'bayesian']
MODELS = ['tabby', 'ctganp', 'icl', 'tabdiff', 'tabdlm', 'tabkg']
TRIALS = [1, 2, 3]

BASE_DIR = "/mnt/data/sonia/mia"
LATEX_OUT = "mia_table.tex"

MODEL_DISPLAY_NAMES = {
    'tabby': 'Tabby',
    'ctganp': 'CTGANP',
    'icl': 'ICL',
    'tabdiff': 'TabDiff',
    'tabdlm': 'TabDLM',
    'tabkg': 'TabKG'
}

def parse_mia_results():
    results = []
    
    for dataset in DATASETS:
        for model in MODELS:
            for trial in TRIALS:
                # TabKG acts as a memorization baseline, so it may only have 1 trial
                if model == 'tabkg' and trial > 1:
                    continue
                    
                file_dir = os.path.join(BASE_DIR, f"{dataset}.{model}.{trial}", "results")
                if not os.path.exists(file_dir): continue
                file_dir = os.path.join(file_dir, os.listdir(file_dir)[0]) # 2 levels of unique name subdirs
                file_path = os.path.join(file_dir, os.listdir(file_dir)[0], 'metrics_E=MIA_detection.csv')
                
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Strip the '%' and cast to float
                        df['auroc'] = df['auroc'].str.rstrip('%').astype(float)
                        
                        # Extract the Worst-Case Adversary (Max AUROC)
                        max_auroc = df['auroc'].max()
                        
                        results.append({
                            'Dataset': dataset,
                            'Model': model,
                            'Trial': trial,
                            'Max_AUROC': max_auroc
                        })
                    except Exception as e:
                        print(f"[-] Error reading {file_path}: {e}")
                else:
                    print(f"[-] Missing MIA file: {file_path}")

    return pd.DataFrame(results)

def generate_latex_table(df):
    # Enforce order
    df['Dataset'] = pd.Categorical(df['Dataset'], categories=DATASETS, ordered=True)
    df['Model'] = pd.Categorical(df['Model'], categories=MODELS, ordered=True)
    
    means = df.groupby(['Dataset', 'Model'], observed=True)['Max_AUROC'].mean()
    stds = df.groupby(['Dataset', 'Model'], observed=True)['Max_AUROC'].std().fillna(0.0)
    
    with open(LATEX_OUT, "w") as f:
        f.write("% ==========================================\n")
        f.write("% Table for Worst-Case MIA AUROC\n")
        f.write("% ==========================================\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        
        col_format = "l" + "c" * len(DATASETS)
        f.write(f"\\begin{{tabular}}{{{col_format}}}\n\\toprule\n")
        
        header = ["Model"] + [d.capitalize() for d in DATASETS]
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        
        # Determine best model per dataset (Closest to 50.0%)
        # Exclude TabKG to prevent highlighting memorization baselines
        best_model_per_dataset = {}
        for dataset in DATASETS:
            if dataset in means.index.get_level_values(0):
                dataset_means = means.loc[dataset].dropna()
                if 'tabkg' in dataset_means.index:
                    dataset_means = dataset_means.drop('tabkg')
                if not dataset_means.empty:
                    # Ideal AUROC is 50% (random guess). We want the minimum distance to 50.0
                    best_model_per_dataset[dataset] = (dataset_means - 50.0).abs().idxmin()

        for model in MODELS:
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            row_data = [display_name]
            
            for dataset in DATASETS:
                try:
                    m = means.loc[(dataset, model)]
                    s = stds.loc[(dataset, model)]
                    is_best = best_model_per_dataset.get(dataset) == model
                    
                    cell_str = f"{m:.1f} \\pm {s:.1f}"
                    if is_best:
                        row_data.append(f"$\\bf{{{cell_str}}}$")
                    else:
                        row_data.append(f"${cell_str}$")
                except KeyError:
                    row_data.append("-")
                    
            f.write(" & ".join(row_data) + " \\\\\n")
            
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Worst-Case Membership Inference Attack (MIA) AUROC. Values closer to $50.0\\%$ indicate optimal privacy preservation (random adversary guessing).}\n")
        f.write("\\label{tab:mia_auroc}\n")
        f.write("\\end{table}\n")
        
    print(f"✅ MIA LaTeX table saved to {LATEX_OUT}")

if __name__ == "__main__":
    df = parse_mia_results()
    if not df.empty:
        generate_latex_table(df)
        