import sys 
import pandas as pd
from typing import Dict, Any
import os
import numpy as np

class BaseFormatter:
    """Base class for translating data to model-specific formats."""
    def format_data(self, df: pd.DataFrame) -> Any:
        raise NotImplementedError
    
    
    def format_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class TabDiffFormatter(BaseFormatter):
    """Flattens nested arrays, applies One-Hot Encoding, and MinMax scaling."""
    def format_data(self, df: pd.DataFrame, meta: Dict[str, Any], split: str):
        datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'models/TabDiff/data', meta['dataset_name'])
        os.makedirs(datadir, exist_ok=True)
        cols = [col for col in meta['columns'] if \
            col in set(meta['categorical'] + meta['integer'] + meta['continuous'] + [meta['target']])]
        df = df[cols] # so this doesn't change relative ordering but removes non categorical text
        df.to_csv(os.path.join(datadir, f'{split}.csv'), index=False)
    
    
    def format_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        import json
        metadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'models/TabDiff/data/Info')
        cols = [col for col in meta['columns'] if \
            col in set(meta['categorical'] + meta['integer'] + meta['continuous'] + [meta['target']])]
        nums = [i for i, col in enumerate(cols) if \
            col in set(meta['continuous'] + meta['integer']) and col != meta['target']]
        cats = [i for i, col in enumerate(cols) if \
            col in set(meta['categorical']) and col != meta['target']]
        targets = [i for i, col in enumerate(cols) if meta['target']==col]
        print(targets)
        td_meta = {
            "name": meta['dataset_name'],
            "task_type": 'regression' if meta['type']=='regression' else 'binclass', # binclass or regression
            "header": "infer",
            "column_names": cols,
            "num_col_idx": nums,  # list of indices of numerical columns
            "cat_col_idx": cats,  # list of indices of categorical columns
            "target_col_idx": targets, # list of indices of the target columns (for MLE)
            "file_type": "csv",
            "data_path": f"data/{meta['dataset_name']}/train.csv",
            "test_path": f"data/{meta['dataset_name']}/test.csv",
            "val_path": f"data/{meta['dataset_name']}/val.csv",
        }
        with open(os.path.join(metadir, f'{meta["dataset_name"]}.json'), 'w') as f:
            json.dump(td_meta, f)
        return td_meta


class LLMFormatter(BaseFormatter):
    """Passes dataframes through directly for k-shot sampling."""
    def format_data(self, df: pd.DataFrame, meta: Dict[str, Any], split: str):
        return df
    
    def format_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        return meta


class TabKGFormatter(BaseFormatter):
    def format_data(self, df: pd.DataFrame, meta: Dict[str, Any], split: str):
        return df
    
    def format_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        return meta


class TabDLMFormatter(BaseFormatter):
    """TabDLM."""
    def format_data(self, df: pd.DataFrame, meta: Dict[str, Any], split: str):
        datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'models/TabDLM/data/tabular', meta['dataset_name'])
        os.makedirs(datadir, exist_ok=True)
        
        # must be num then cat then text (target wherever)
        df = df[meta['continuous'] + meta['integer'] + meta['categorical'] + meta['text']]
        
        df.to_csv(os.path.join(datadir, f'{split}.csv'), index=False)
        return df
    
    
    def format_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        meta['nums'] = meta['continuous'] + meta['integer']
        meta['columns'] = meta['continuous'] + meta['integer'] + meta['categorical'] + meta['text']
        meta['type'] = 'binclass' if meta['type'] == 'classification' else 'regression'
        return meta

        
class TabbyFormatter(BaseFormatter):
    """Tabby."""
    def format_data(self, df: pd.DataFrame, meta: Dict[str, Any], split: str):
        datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'models/tabby/data', meta['dataset_name'], 'latest')
        os.makedirs(datadir, exist_ok=True)
        df.to_csv(os.path.join(datadir, f'{split}.csv'), index=False)
    
    
    def format_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        import json
        datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'models/tabby/data', meta['dataset_name'], 'latest')
        os.makedirs(datadir, exist_ok=True)
        
        ords = [col for col in meta['columns'] if col in \
            set(meta['categorical']) and col != meta['target']]
        nums = [col for col in meta['columns'] if col in \
            set(meta['continuous'] + meta['integer']) and col != meta['target']]
        
        tabby_meta = {
            'dataset_name': meta['dataset_name'],
            'task': meta['type'],
            'creation_time': 'latest',
            'max_col_length': 50,
            'cols': meta['columns'],
            'ords': ords,
            'nums': nums,
            'target': [meta['target']]
        }
        with open(os.path.join(datadir, 'config.json'), 'w') as f:
            json.dump(tabby_meta, f)
        return tabby_meta


class GANFormatter(BaseFormatter):
    """CTGAN+"""
    def format_data(self, df: pd.DataFrame, meta: Dict[str, Any], split: str):
        all_valid_cols = meta.get('categorical', []) + meta.get('continuous', []) + meta.get('integer', []) + meta.get('datetime', [])
        print(f"dropping columns {set(df.columns)-set(all_valid_cols)} due to GAN limitation")
        df = df[all_valid_cols].copy() # prevent SettingWithCopyWarning
        
        if len(meta.get('datetime', [])) > 0:
            for col in meta['datetime']:
                # temporarily fill NaT so astype('int64') doesn't underflow
                is_missing = df[col].isna()
                # Fill, cast to UNIX epoch, then convert back to float so we can use np.nan
                df[col] = df[col].fillna(pd.Timestamp("1970-01-01")).astype('int64') // 10**9
                df[col] = df[col].astype(float)
                # put 0s where the NaTs were
                df.loc[is_missing, col] = 0.0
                
        print(df.shape)
        return df
    
    
    # must be called prior to format_data
    def format_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        meta['type'] = meta['type'].title()
        meta.pop('text', None)
        meta['continuous'] = meta.get('continuous', []) + meta.pop('datetime', [])
        return meta



class UnifiedDataLoader:
    """The main orchestrator for the benchmark pipeline."""
    
    
    DATASETS = ["honeypot", "stroke", "cern", "lob", "moma", "olist", 'bayesian']
    MODEL_TYPES = ["tabdiff", "llm", "tabdlm", "tabby", "gan", "tabkg"]


    def __init__(self, dataset_name: str, target_model_type: str):
        assert dataset_name in self.DATASETS, f"Dataset {dataset_name} not supported."
        assert target_model_type in self.MODEL_TYPES, f"Model type {target_model_type} not supported."
        
        self.dataset_name = dataset_name
        self.target_model_type = target_model_type
        
        # 1. Load raw data
        self.raw_train, self.raw_val, self.raw_test, self.raw_all = self._load_raw_data()
        
        # 2. Attach downstream task metadata
        self.meta = self._get_task_metadata()
        
        # 3. Initialize the correct formatter
        self.formatter = self._initialize_formatter()
            

    def _load_raw_data(self) -> pd.DataFrame:
        """Loads cached data or triggers the specific parsing script."""
        std_train_frac = 0.75 
        std_val_frac = 0.10
        # std_test_frac is the rest 0.15
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, 'data/processed', self.dataset_name)
        
        if os.path.exists(os.path.join(data_dir, 'train.csv')) and \
                    os.path.exists(os.path.join(data_dir, 'val.csv')) and \
                    os.path.exists(os.path.join(data_dir, 'test.csv')) and \
                    os.path.exists(os.path.join(data_dir, 'all.csv')):
            df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
            df_val = pd.read_csv(os.path.join(data_dir, 'val.csv'))
            df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
            df = pd.read_csv(os.path.join(data_dir, 'all.csv'))
                
        else:
            if self.dataset_name == 'honeypot':
                import numpy as np
                import json
                file_path = os.path.join(base_dir, 'data/raw/honeypot.json')
                
                print("Loading SecRepo JSONL data...")
                df = pd.read_json(file_path, lines=True)
                
                print("Flattening nested payload...")
                def safe_parse(val):
                    try:
                        return json.loads(val)
                    except:
                        return {}
                
                # Expand the parsed JSON dictionary into individual columns
                payload_df = pd.json_normalize(df['payload'].apply(safe_parse))
                df = pd.concat([df, payload_df], axis=1)
                df.replace(['NaN', 'nan', 'None', 'null', '', ' '], np.nan, inplace=True)
                
                # Split the mixed-type 'source' array into distinct IP and Port columns
                df['source_ip'] = df['source'].apply(lambda x: str(x[0]) if isinstance(x, list) and len(x) > 0 else "Missing")
                df['source_port'] = df['source'].apply(lambda x: str(x[1]) if isinstance(x, list) and len(x) > 1 else "Missing")
                
                # Sort chronologically to prevent data leakage in the ML evaluator
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                df = df.sort_values('time').reset_index(drop=True)
                
                # Clean up the metadata and original nested structures
                cols_to_drop = ['_id', 'ident', 'normalized', 'channel', 'payload', 'source', 'timestamp', 'time', 'filename']
                df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
                
                # Drop rows where the target variable is completely missing
                df.dropna(subset=['pattern'], inplace=True)
                
                cols = ['pattern', 'source_ip', 'source_port', 'request_raw', 'request_url', 'attackerPort', 'victimPort', 'md5']
                df = df[cols]
                
                # Chronological split
                size = len(df)
                df_train = df[:int(std_train_frac*size)]
                df_val = df[int(std_train_frac*size):int((std_train_frac+std_val_frac)*size)]
                df_test = df[int((std_train_frac+std_val_frac)*size):]
                
            elif self.dataset_name == 'stroke':
                import kagglehub
                
                print("Pulling Stroke Prediction Medical dataset...")
                path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
                
                df = pd.read_csv(os.path.join(path, "healthcare-dataset-stroke-data.csv"))
                
                # Drop the meaningless patient ID to prevent memorization
                df.drop(columns=['id'], inplace=True)
                
                # The 'bmi' column contains "N/A" strings instead of real NaNs.
                # We force them to np.nan so your _force_dtypes function handles them correctly.
                df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
                
                # Drop rows where the target itself is missing (though rare here)
                df.dropna(subset=['stroke'], inplace=True)
                
                # This dataset is static (no temporal component), so we shuffle it 
                # before splitting to ensure even class distribution.
                df = df.sample(random_state=42, frac=1.0).reset_index(drop=True)
                
                size = len(df)
                df_train = df[:int(std_train_frac*size)]
                df_val = df[int(std_train_frac*size):int((std_train_frac+std_val_frac)*size)]
                df_test = df[int((std_train_frac+std_val_frac)*size):]
                
            elif self.dataset_name == 'cern':
                import kagglehub
                
                print("Pulling CERN Electron Collision dataset...")
                path = kagglehub.dataset_download("fedesoriano/cern-electron-collision-data")
                
                df = pd.read_csv(os.path.join(path, "dielectron.csv"))
                
                # arbitrary identifiers 
                df.drop(columns=['Run', 'Event'], inplace=True, errors='ignore')
                df.dropna(inplace=True)
                
                df.columns = [name.strip()+'_' for name in df.columns]
                
                # downstream regressor will predict mass M
                df = df.sample(random_state=42, frac=1.0).reset_index(drop=True)
                
                size = len(df)
                df_train = df[:int(std_train_frac*size)]
                df_val = df[int(std_train_frac*size):int((std_train_frac+std_val_frac)*size)]
                df_test = df[int((std_train_frac+std_val_frac)*size):]
                
            elif self.dataset_name == 'lob':
                import kagglehub
                gap = 120 # 2 hours to prevent leakage
                df = kagglehub.dataset_load(
                    kagglehub.KaggleDatasetAdapter.PANDAS,
                    "martinsn/high-frequency-crypto-limit-order-book-data",
                    "BTC_1min.csv"
                )
                df.drop(columns=['Unnamed: 0'], inplace=True)
                df = df.sort_values('system_time').reset_index(drop=True) # just in case it isnt in order
                
                core_columns = ['system_time', 'midpoint']
    
                max_depth=5
                for i in range(max_depth):
                    core_columns.extend([
                        f'bids_distance_{i}', 
                        f'asks_distance_{i}', 
                        f'bids_notional_{i}', 
                        f'asks_notional_{i}'
                    ])
                    
                df = df[core_columns]
                df['system_time'] = pd.to_datetime(df['system_time'])
                df['system_time'] = df['system_time'].dt.strftime('%Y-%m-%d %H:%M')

                # making the regression target
                k_horizon = 5

                # shift(-5) pulls the price from 5 rows (minutes) in the future up to the current row
                df['future_return'] = (df['midpoint'].shift(-k_horizon) - df['midpoint']) / df['midpoint']

                # Drop the last 5 rows because they now have NaNs for the future return
                df.dropna(subset=['future_return'], inplace=True)
                
                size = len(df) - 2*gap
                df_train = df[:int(std_train_frac*size)]
                df_val = df[int(std_train_frac*size) + gap:int((std_train_frac+std_val_frac)*size + gap)]
                df_test = df[int((std_train_frac+std_val_frac)*size + 2*gap):]
                
                cols_to_drop = ['system_time', 'midpoint']
                df_train.drop(columns=cols_to_drop, inplace=True)
                df_val.drop(columns=cols_to_drop, inplace=True)
                df_test.drop(columns=cols_to_drop, inplace=True)
                df.drop(columns=cols_to_drop, inplace=True)

                print(df_train.shape, df_val.shape, df_test.shape)
            
            elif self.dataset_name == 'moma':
                df = pd.read_csv(os.path.join(base_dir, 'data/raw/moma/collection/Artworks.csv'))
                df.drop_duplicates(subset=['Title'], keep='first', inplace=True)
                df.drop_duplicates(subset=['Artist'], keep='first', inplace=True)
                # classification is too similar to department
                df.drop(columns=['URL', 'ImageURL', 'Classification', 'Artist', 'Seat Height (cm)'], inplace=True)
                df = df.replace(r'[\r\n]+', ' ', regex=True)
                df = df.sample(random_state=42, frac=1.0).reset_index(drop=True)
                df_train = df[:int(std_train_frac*len(df))]
                df_val = df[int(std_train_frac*len(df)):int((std_train_frac+std_val_frac)*len(df))]
                df_test = df[int((std_train_frac+std_val_frac)*len(df)):]
            
            elif self.dataset_name == 'olist':
                import kagglehub
                
                print("Pulling Olist E-Commerce dataset...")
                path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
                
                # Olist is relational, so we read and flatten the core tables
                orders = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"))
                items = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"))
                products = pd.read_csv(os.path.join(path, "olist_products_dataset.csv"))
                reviews = pd.read_csv(os.path.join(path, "olist_order_reviews_dataset.csv"))
                
                print("Joining tables into a flat commerce schema...")
                df = orders.merge(items, on='order_id', how='inner')
                df = df.merge(products, on='product_id', how='left')
                df = df.merge(reviews, on='order_id', how='left')
                
                df['product_name_length'] = df['product_name_lenght'] # typo in original data
                df['product_description_length'] = df['product_description_lenght']
                
                cols_to_keep = [
                    'review_score', 'price', 'freight_value', 
                    'product_category_name', 'product_name_length', 
                    'product_description_length', 'product_photos_qty',
                    'order_status', 'order_purchase_timestamp', 
                    'order_delivered_customer_date'
                ]
                df = df[cols_to_keep].dropna(subset=['freight_value'])
                
                # Chronological split to prevent temporal leakage
                df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
                df = df.sort_values('order_purchase_timestamp').reset_index(drop=True)
                
                size = len(df)
                df_train = df[:int(std_train_frac*size)]
                df_val = df[int(std_train_frac*size):int((std_train_frac+std_val_frac)*size)]
                df_test = df[int((std_train_frac+std_val_frac)*size):]
            
            elif self.dataset_name == 'bayesian':
                import numpy as np
                
                print("Generating Pure Synthetic DAG (The Collider Trap)...")
                # Set a strict seed so the benchmark is 100% reproducible
                np.random.seed(42)
                n_samples = 100000
                
                # 1. Independent Roots
                x1 = np.random.normal(0, 1, n_samples)
                x2 = np.random.normal(0, 1, n_samples)
                
                # 2. The Collider (X1 -> X3 <- X2)
                # X1 and X2 are independent, but highly correlated given X3
                x3 = x1 + x2 + np.random.normal(0, 0.1, n_samples)
                
                # 3. The Confounder (Drives both X5 and the Target)
                x4_hidden = np.random.uniform(-3, 3, n_samples)
                
                # 4. The Non-Linear Proxy
                # Models must learn a sine wave manifold, not just a Gaussian
                x5 = np.sin(x4_hidden) + np.random.normal(0, 0.05, n_samples)
                
                # 5. The Categorical Noise
                # A purely random string column to test if models get distracted
                categories = ['Alpha', 'Beta', 'Gamma', 'Delta']
                x6_cat = np.random.choice(categories, n_samples)
                
                # 6. The Target (Strict Logical Boundary)
                # To get a 1, X3 must be positive AND the hidden confounder must be positive
                prob = 1 / (1 + np.exp(-(x3 + x4_hidden)))
                target = (prob > 0.5).astype(int)
                
                df = pd.DataFrame({
                    'root_1': x1,
                    'root_2': x2,
                    'collider': x3,
                    'proxy_sin': x5,
                    'distractor_cat': x6_cat,
                    'target_class': target
                })
                
                # Because it's synthetic and math-based, order doesn't matter
                size = len(df)
                df_train = df[:int(std_train_frac*size)]
                df_val = df[int(std_train_frac*size):int((std_train_frac+std_val_frac)*size)]
                df_test = df[int((std_train_frac+std_val_frac)*size):]
                
            else:
                raise NotImplementedError(f"Loading for {self.dataset_name} not yet implemented")
            
            os.makedirs(data_dir, exist_ok=True)
            df_train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
            df_val.to_csv(os.path.join(data_dir, 'val.csv'), index=False)
            df_test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
            df.to_csv(os.path.join(data_dir, 'all.csv'), index=False)
            
        return df_train, df_val, df_test, df


    def _force_dtypes(self, df, meta):
        # Force all text and categorical columns to strictly be strings
        for col in meta['categorical'] + meta['text']:
            # Filling NaNs with a string placeholder prevents them from being read as float NaNs
            df[col] = df[col].fillna("Missing").astype(str)

        # Force all continuous and integer columns to strictly be numeric
        for col in meta['continuous'] + meta['integer']:
            # errors='coerce' turns any hidden text in numeric columns into float NaNs
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # CTAB-GAN+ needs a numeric value to calculate min/max. 
            # Fill any coerced NaNs with 0 so the transformer doesn't crash.
            df[col] = df[col].fillna(0)
            
        return df


    def _get_task_metadata(self) -> Dict:
        """Returns the target column and task type (e.g., classification, regression)."""
        tasks = {
            "honeypot":         {"target": 'pattern', "type": "classification"},
            "stroke":           {"target": "stroke", "type": "classification"}, 
            "cern":             {"target": "M_", "type": "regression"},
            "lob":              {"target": "future_return", "type": "regression"},
            "moma":             {"target": "Department", "type": "classification"},
            "olist":            {"target": "freight_value", "type": "regression"},
            "bayesian":         {"target": "target_class", "type": "classification"},
            # "nexrad":           {"target": "is_severe_hail", "type": "classification"},
        }
        meta = tasks[self.dataset_name]
        meta['dataset_name'] = self.dataset_name
        meta['columns'] = list(self.raw_train.columns) # this is in order!

        meta.update({
            "categorical": [],
            "integer": [],
            "continuous": [], # Maps to "general_columns" in CTAB-GAN+
            "text": [],       # Maps to "non_categorical_columns" in CTAB-GAN+
            "datetime": []
        })
        
        total_rows = len(self.raw_train)
        
        for col in self.raw_train.columns:
            # 1. Drop NaNs just for the inference check
            series = self.raw_train[col].dropna()
            if len(series) == 0:
                continue
                
            unique_count = series.nunique()
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_datetime = pd.api.types.is_datetime64_any_dtype(series)
            
            # --- Datetime Detection ---
            if is_datetime:
                meta["datetime"].append(col)
                continue
                
            # Try to aggressively cast strings to datetime to catch hidden dates
            if series.dtype == 'object':
                try:
                    # Only check the first few rows to save time
                    pd.to_datetime(series.iloc[:10], errors='raise')
                    meta["datetime"].append(col)
                    continue
                except (ValueError, TypeError):
                    pass
            
            # --- Numeric Detection (Integer vs Continuous) ---
            max_categorical_threshold = 200
            categorical_ratio = 0.1
            if is_numeric:
                # Check if all numbers are whole numbers (even if stored as float)
                if (series % 1 == 0).all():
                    # is it actually a categorical ID code? (e.g., Status = 1, 2, 3)
                    if unique_count <= max_categorical_threshold or (unique_count / total_rows) < categorical_ratio:
                        meta["categorical"].append(col)
                    else:
                        meta["integer"].append(col)
                else:
                    meta["continuous"].append(col)
                continue
                
            # --- String / Object Detection (Categorical vs Free Text) ---
            # It's an object/string. We use cardinality to decide if it's a category or free text.
            if unique_count <= max_categorical_threshold or (unique_count / total_rows) < categorical_ratio:
                meta["categorical"].append(col)
            else:
                meta["text"].append(col)
                
                
        self.raw_train = self._force_dtypes(self.raw_train, meta)
        self.raw_val = self._force_dtypes(self.raw_val, meta)
        self.raw_test = self._force_dtypes(self.raw_test, meta)

        return meta


    def _initialize_formatter(self) -> BaseFormatter:
        if self.target_model_type == 'tabdiff':
            return TabDiffFormatter()
        elif self.target_model_type == "gan":
            return GANFormatter()
        elif self.target_model_type == "llm":
            return LLMFormatter()
        elif self.target_model_type == "tabdlm":
            return TabDLMFormatter()
        elif self.target_model_type == "tabby":
            return TabbyFormatter()
        elif self.target_model_type == 'tabkg':
            return TabKGFormatter()


    def get_train_data(self):
        return self.formatter.format_data(self.raw_train, self.meta, 'train')
    
    
    def get_val_data(self):
        return self.formatter.format_data(self.raw_val, self.meta, 'val')
    
    
    def get_test_data(self):
        return self.formatter.format_data(self.raw_test, self.meta, 'test')
    
    
    def get_all_data(self):
        return self.formatter.format_data(self.raw_test, self.meta, 'all')
    
    
    def get_metadata(self):
        return self.formatter.format_metadata(self.meta)
    
    
if __name__ == "__main__":
    # this is here for debugging 
    loader = UnifiedDataLoader(dataset_name=sys.argv[-1], target_model_type="llm")
    print(loader.raw_train.shape, loader.raw_train.head(), loader.raw_train.dtypes, sep='\n')
    