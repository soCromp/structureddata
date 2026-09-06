import numpy as np
import pandas as pd
import torch
import warnings
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

# Initialize the global text encoder 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def evaluate_mle(synth_train, real_test, real_train_reference, meta):
    """
    Train on Synthetic, Test on Real (TSTR).
    Transforms dates into epochs, free-text into semantic embeddings, and 
    numerical/categorical features into standard scaled representations.
    """
    target_col = meta.get('target')
    task_type = meta.get('type', 'classification').lower()
    
    # 1. Feature Identification
    categorical_cols = [c for c in meta.get('categorical', []) if c != target_col and c in synth_train.columns]
    continuous_cols = [c for c in meta.get('continuous', []) + meta.get('integer', []) if c != target_col and c in synth_train.columns]
    text_cols = [c for c in meta.get('text', []) if c != target_col and c in synth_train.columns]
    datetime_cols = [c for c in meta.get('datetime', []) if c != target_col and c in synth_train.columns]
    
    # Fast-fail if mode collapse destroyed the target column
    if target_col not in synth_train.columns or len(synth_train) < 10:
        return _get_empty_results(task_type)

    # 2. Target Encoding
    y_synth = synth_train[target_col].copy()
    y_test = real_test[target_col].copy()
    
    if 'class' in task_type:
        le_target = LabelEncoder()
        # Fit on union to avoid crashing on hallucinated classes
        le_target.fit(pd.concat([real_train_reference[target_col], y_synth, y_test]).astype(str))
        y_synth = le_target.transform(y_synth.astype(str))
        y_test = le_target.transform(y_test.astype(str))
    else:
        y_synth = pd.to_numeric(y_synth, errors='coerce').fillna(0).values
        y_test = pd.to_numeric(y_test, errors='coerce').fillna(0).values

    # 3. Datetime to Epoch Conversion
    # We map dates to Unix timestamps, then treat them as continuous variables for scaling
    for df in [synth_train, real_test, real_train_reference]:
        for col in datetime_cols:
            series = pd.to_datetime(df[col], errors='coerce')
            is_missing = series.isna()
            df[col] = series.fillna(pd.Timestamp("1970-01-01")).astype('int64') // 10**9
            df[col] = df[col].astype(float)
            df.loc[is_missing, col] = 0.0
            
    continuous_cols.extend(datetime_cols)

    # 4. Multimodal Feature Processing
    def extract_features(df, is_synth=False):
        matrices = []
        
        # Text Embeddings
        for col in text_cols:
            texts = df[col].fillna("Missing").astype(str).tolist()
            with torch.no_grad():
                embs = text_encoder.encode(texts, convert_to_numpy=True)
            matrices.append(embs)
            
        # Categorical (Label Encoded)
        for col in categorical_cols:
            le = LabelEncoder()
            # Fit strictly on the union to ensure aligned integer mapping
            union = pd.concat([real_train_reference[col], synth_train[col], real_test[col]]).fillna("Missing").astype(str)
            le.fit(union)
            encoded = le.transform(df[col].fillna("Missing").astype(str)).reshape(-1, 1)
            matrices.append(encoded)
            
        # Continuous (Scaled & Clipped)
        if continuous_cols:
            r_ref = real_train_reference[continuous_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
            target_vals = df[continuous_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
            
            scaler = StandardScaler()
            scaler.fit(r_ref) # Fit strictly on REAL data to capture how far synthetic data drifts
            scaled = scaler.transform(target_vals)
            
            # Clip astronomical floats (hallucinations) to prevent Random Forest float32 overflows
            if is_synth:
                scaled = np.clip(scaled, -1e15, 1e15)
                
            matrices.append(scaled)
            
        return np.hstack(matrices) if matrices else np.empty((len(df), 0))

    X_synth = extract_features(synth_train, is_synth=True)
    X_test = extract_features(real_test, is_synth=False)

    if X_synth.shape[1] == 0:
        return _get_empty_results(task_type)

    # 5. Downstream Evaluation (Train on Synthetic, Test on Real)
    try:
        if 'class' in task_type:
            # 1. Fast-fail if the synthetic data completely mode-collapsed to a single class
            if len(np.unique(y_synth)) < 2:
                print("  [!] Target mode collapse in synthetic data. Cannot train classifier.")
                return {"MLE_Accuracy": np.nan, "MLE_F1": np.nan, "MLE_AUC": np.nan}
                
            clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
            clf.fit(X_synth, y_synth)
            
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)
            
            # 2. Probability Padding: Reconstruct the full probability matrix
            # so it matches the real test set, even if the synthetic data dropped classes.
            total_classes = len(le_target.classes_)
            full_probs = np.zeros((len(y_test), total_classes))
            
            # Map the columns the classifier DID learn to their correct index
            for i, class_idx in enumerate(clf.classes_):
                full_probs[:, class_idx] = probs[:, i]
                
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')
            
            # 3. Safe AUC calculation using the padded matrix
            try:
                if total_classes == 2:
                    auc = roc_auc_score(y_test, full_probs[:, 1])
                else:
                    auc = roc_auc_score(y_test, full_probs, multi_class='ovr', labels=np.arange(total_classes))
            except Exception as e:
                print(f"  [!] AUC calculation failed (likely missing classes in test split): {e}")
                auc = np.nan
                
            return {"MLE_Accuracy": acc, "MLE_F1": f1, "MLE_AUC": auc}
            
        else:
            # Regression is immune to class-count mismatches
            reg = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
            reg.fit(X_synth, y_synth)
            
            preds = reg.predict(X_test)
            
            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            
            return {"MLE_R2": r2, "MLE_RMSE": rmse, "MLE_MAE": mae}
            
    except Exception as e:
        print(f"  [!] Downstream MLE evaluation failed: {e}")
        return _get_empty_results(task_type)

def _get_empty_results(task_type):
    if 'class' in task_type:
        return {"MLE_Accuracy": np.nan, "MLE_F1": np.nan, "MLE_AUC": np.nan}
    return {"MLE_R2": np.nan, "MLE_RMSE": np.nan, "MLE_MAE": np.nan}
