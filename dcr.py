import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

def evaluate_multimodal_dcr(real_train_df, synth_df, real_test_df, categorical_cols, continuous_cols, text_cols):
    """
    Computes the Distance to Closest Record (DCR) in a unified multimodal vector space.
    Returns the mean DCR for both the synthetic data and the real test baseline.
    """
    # 1. Align schemas and handle missing values
    common_cols = [c for c in real_train_df.columns if c in synth_df.columns and c in real_test_df.columns]
    
    cat_eval = [c for c in categorical_cols if c in common_cols and c not in text_cols]
    cont_eval = [c for c in continuous_cols if c in common_cols and c not in text_cols]
    text_eval = [c for c in text_cols if c in common_cols]

    def prep_df(df):
        out = df[common_cols].copy()
        for c in cat_eval:
            out[c] = out[c].fillna("Missing").astype(str)
        for c in cont_eval:
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0)
        for c in text_eval:
            out[c] = out[c].fillna("").astype(str)
        return out

    train_clean = prep_df(real_train_df)
    synth_clean = prep_df(synth_df)
    test_clean = prep_df(real_test_df)

    # Fast-fail for extreme mode collapse
    if len(synth_clean) < 10 or len(train_clean) < 10:
        return {"DCR_Synth": np.nan, "DCR_Baseline": np.nan}

    # 2. Vectorize Tabular Features (StandardScaler for continuous, OHE for categorical)
    transformers = []
    if cont_eval:
        transformers.append(('num', StandardScaler(), cont_eval))
    if cat_eval:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_eval))
    
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    
    # Fit tabular processor strictly on training data
    train_tab_vec = preprocessor.fit_transform(train_clean)
    synth_tab_vec = preprocessor.transform(synth_clean)
    test_tab_vec = preprocessor.transform(test_clean)

    # 3. Vectorize Text Features
    train_text_vecs, synth_text_vecs, test_text_vecs = [], [], []
    if text_eval:
        # MiniLM is fast and outputs normalized 384-dimensional vectors
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        for col in text_eval:
            train_text_vecs.append(embedder.encode(train_clean[col].tolist(), show_progress_bar=False))
            synth_text_vecs.append(embedder.encode(synth_clean[col].tolist(), show_progress_bar=False))
            test_text_vecs.append(embedder.encode(test_clean[col].tolist(), show_progress_bar=False))

    # 4. Concatenate Modalities into a single feature space
    def build_final_matrix(tab_vec, text_vec_list):
        arrays = [tab_vec] + text_vec_list if len(tab_vec) > 0 else text_vec_list
        if not arrays:
            return np.array([])
        return np.hstack(arrays)

    X_train = build_final_matrix(train_tab_vec, train_text_vecs)
    X_synth = build_final_matrix(synth_tab_vec, synth_text_vecs)
    X_test = build_final_matrix(test_tab_vec, test_text_vecs)

    if X_train.size == 0:
        return {"DCR_Synth": np.nan, "DCR_Baseline": np.nan}

    # 5. Compute Exact Nearest Neighbors (Euclidean L2 Distance)
    # Using n_neighbors=1 to find the absolute closest record in the training set
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='auto')
    nn.fit(X_train)

    # DCR for Synthetic Data
    dcr_synth_distances, _ = nn.kneighbors(X_synth)
    mean_dcr_synth = float(np.mean(dcr_synth_distances))

    # DCR for Test Data (The 'Goldilocks' Baseline)
    # We use n_neighbors=2 for test vs train ONLY if test contains training rows, 
    # but since it's a holdout set, n_neighbors=1 is correct.
    dcr_test_distances, _ = nn.kneighbors(X_test)
    mean_dcr_baseline = float(np.mean(dcr_test_distances))

    return {
        "DCR_Synth": mean_dcr_synth,
        "DCR_Baseline": mean_dcr_baseline
    }
    