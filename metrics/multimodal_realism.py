import numpy as np
import pandas as pd
import torch
from scipy.linalg import sqrtm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
from sentence_transformers import SentenceTransformer

# Initialize the global encoder 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def compute_frechet_distance(real_features, synth_features):
    """Computes the Fréchet Distance between two feature distributions."""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = synth_features.mean(axis=0), np.cov(synth_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)

def evaluate_multimodal_realism(real_df, synth_df, categorical_cols, continuous_cols, sample_size=5000):
    """
    Computes the Classifier Two-Sample Test (C2ST) and Joint-FID across 
    the full concatenated feature space (text embeddings + normalized tabular).
    """
    synth_df[continuous_cols] = synth_df[continuous_cols].apply(pd.to_numeric, errors='coerce')
    synth_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    synth_df = synth_df.dropna(subset=continuous_cols)
        
    # Align rows to a max sample size to prevent OOM errors and speed up the Random Forest
    n_samples = min(len(real_df), len(synth_df), sample_size)
    
    if n_samples < 100:  # Prevent crashes on severe mode collapse
        return {"Joint_FID": np.nan, "C2ST_Accuracy": np.nan, "C2ST_AUC": np.nan}
        
    real_sub = real_df.sample(n_samples, random_state=42).copy()
    synth_sub = synth_df.sample(n_samples, random_state=42).copy()
    
    
    text_cols = [c for c in ['Title', 'request_raw'] if c in real_sub.columns]
    
    real_matrices = []
    synth_matrices = []
    
    # 1. Process Text (Embeddings)
    if text_cols:
        for col in text_cols:
            real_text = real_sub[col].fillna("Missing").astype(str).tolist()
            synth_text = synth_sub[col].fillna("Missing").astype(str).tolist()
            
            with torch.no_grad():
                real_embs = text_encoder.encode(real_text, convert_to_numpy=True)
                synth_embs = text_encoder.encode(synth_text, convert_to_numpy=True)
                
            real_matrices.append(real_embs)
            synth_matrices.append(synth_embs)
            
    # 2. Process Tabular (Numerical & Categorical)
    eval_cats = [c for c in categorical_cols if c in real_sub.columns and c not in text_cols]
    eval_conts = [c for c in continuous_cols if c in real_sub.columns and c not in text_cols]
    
    if eval_cats:
        for col in eval_cats:
            # 1. Fill NaNs FIRST, then convert to string for both splits
            r_series = real_sub[col].fillna("Missing").astype(str)
            s_series = synth_sub[col].fillna("Missing").astype(str)
            
            # 2. Combine the already-sanitized series so classes are identical
            combined = pd.concat([r_series, s_series], axis=0)
            
            le = LabelEncoder()
            le.fit(combined)
            
            r_cat = le.transform(r_series).reshape(-1, 1)
            s_cat = le.transform(s_series).reshape(-1, 1)
            
            real_matrices.append(r_cat)
            synth_matrices.append(s_cat)
            
    if eval_conts:
        scaler = StandardScaler()
        
        # 1. Safely parse numbers, explicitly converting string 'inf' or np.inf to NaN, then to 0
        r_cont = real_sub[eval_conts].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors='coerce').fillna(0).values
        s_cont = synth_sub[eval_conts].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors='coerce').fillna(0).values
        
        scaler.fit(r_cont)
        scaled_r = scaler.transform(r_cont)
        scaled_s = scaler.transform(s_cont)
        
        # 2. Clip astronomical outliers to prevent float32 overflow in Random Forest
        # 1e15 is safely under the 3.4e38 limit, but large enough to correctly ruin the model's score
        float32_safe_max = 1e15
        scaled_s = np.clip(scaled_s, -float32_safe_max, float32_safe_max)
        scaled_r = np.clip(scaled_r, -float32_safe_max, float32_safe_max)
        
        real_matrices.append(scaled_r)
        synth_matrices.append(scaled_s)
        
    # 3. Concatenate into Joint Multimodal Space
    if not real_matrices:
        return {"Joint_FID": np.nan, "C2ST_Accuracy": np.nan, "C2ST_AUC": np.nan}
        
    X_real_joint = np.hstack(real_matrices)
    X_synth_joint = np.hstack(synth_matrices)
    
    # 4. Joint-FID
    try:
        joint_fid = compute_frechet_distance(X_real_joint, X_synth_joint)
    except Exception as e:
        print(f"  [!] Joint-FID failed: {e}")
        joint_fid = np.nan
        
    # 5. Classifier Two-Sample Test (C2ST)
    try:
        # Create dataset: Real = 1, Synth = 0
        X_combined = np.vstack([X_real_joint, X_synth_joint])
        y_combined = np.array([1] * n_samples + [0] * n_samples)
        
        X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
        
        # Random Forest is highly robust to the mixed scaling between dense embeddings and encoded categoricals
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        
        c2st_acc = accuracy_score(y_test, preds)
        c2st_auc = roc_auc_score(y_test, probs)
    except Exception as e:
        print(f"  [!] C2ST failed: {e}")
        c2st_acc, c2st_auc = np.nan, np.nan
        
    return {
        "Joint_FID": joint_fid,
        "C2ST_Accuracy": c2st_acc, 
        "C2ST_AUC": c2st_auc
    }