import pandas as pd
from typing import Literal, Dict, Any
import os

class BaseFormatter:
    """Base class for translating data to model-specific formats."""
    def format_train_data(self, df: pd.DataFrame) -> Any:
        raise NotImplementedError

class DiffusionFormatter(BaseFormatter):
    """Flattens nested arrays, applies One-Hot Encoding, and MinMax scaling."""
    def format_train_data(self, df: pd.DataFrame):
        # 1. Flatten arrays (e.g. OpenCorporates officers -> officer_1, officer_2)
        # 2. Drop high-cardinality strings diffusion can't handle
        # 3. Return a purely numerical numpy array/tensor
        pass

class LLMFormatter(BaseFormatter):
    """Serializes rows into text strings for autoregressive models (GReaT, ICL)."""
    def format_train_data(self, df: pd.DataFrame):
        # E.g., "The company name is ACME. The incorporation date is 2014-07-15."
        pass

class TabDLMFormatter(BaseFormatter):
    """Separates text columns and numerical columns for joint diffusion."""
    def format_train_data(self, df: pd.DataFrame):
        # Returns Dict: {"text_inputs": [...], "continuous_inputs": [...]}
        pass
        
class TabbyFormatter(BaseFormatter):
    """Custom serialization for Tabby's column-specific MoE."""
    def format_train_data(self, df: pd.DataFrame):
        pass

class UnifiedDataLoader:
    """The main orchestrator for the benchmark pipeline."""
    
    DATASETS = ["secrepo", "caida", "clinicaltrials", "nexrad", "crypto_lob", "opencorporates", "moma"]
    MODEL_TYPES = ["diffusion", "llm", "tabdlm", "tabby", "gan"]

    def __init__(self, dataset_name: str, target_model_type: str):
        assert dataset_name in self.DATASETS, f"Dataset {dataset_name} not supported."
        assert target_model_type in self.MODEL_TYPES, f"Model type {target_model_type} not supported."
        
        self.dataset_name = dataset_name
        self.target_model_type = target_model_type
        
        # 1. Load raw data
        self.raw_train, self.raw_val, self.raw_test = self._load_raw_data()
        
        # 2. Attach downstream task metadata
        self.task_meta = self._get_task_metadata()
        
        # 3. Initialize the correct formatter
        self.formatter = self._initialize_formatter()

    def _load_raw_data(self) -> pd.DataFrame:
        """Loads cached data or triggers the specific parsing script."""
        std_train_frac = 0.75 
        std_val_frac = 0.10
        # std_test_frac is the rest 0.15
        
        if self.dataset_name == 'moma':
            if os.path.exists('data/processed/moma/train.csv') and \
                        os.path.exists('data/processed/moma/val.csv') and \
                        os.path.exists('data/processed/moma/test.csv'):
                df_train = pd.read_csv('data/processed/moma/train.csv')
                df_val = pd.read_csv('data/processed/moma/val.csv')
                df_test = pd.read_csv('data/processed/moma/test.csv')
            else:
                df = pd.read_csv('data/raw/moma/collection/Artworks.csv')
                df.drop_duplicates(subset=['Title'], keep='first', inplace=True)
                df.drop_duplicates(subset=['Artist'], keep='first', inplace=True)
                # classification is too similar to department
                df.drop(columns=['URL', 'ImageURL', 'Classification', 'Artist', 'Seat Height (cm)'], inplace=True)
                df = df.sample(random_state=42, frac=1.0).reset_index(drop=True)
                df_train = df[:int(std_train_frac*len(df))]
                df_val = df[int(std_train_frac*len(df)):int((std_train_frac+std_val_frac)*len(df))]
                df_test = df[int(std_train_frac+std_val_frac)*len(df):]
                os.makedirs('data/processed/moma/', exist_ok=True)
                df_train.to_csv('data/processed/moma/train.csv', index=False)
                df_val.to_csv('data/processed/moma/val.csv', index=False)
                df_test.to_csv('data/processed/moma/test.csv', index=False)
        else:
            raise NotImplementedError(f"Loading for {self.dataset_name} not yet implemented")
            
        return df_train, df_val, df_test

    def _get_task_metadata(self) -> Dict:
        """Returns the target column and task type (e.g., classification, regression)."""
        tasks = {
            "crypto_lob":       {"target": "midpoint_direction", "type": "classification"},
            "nexrad":           {"target": "is_severe_hail", "type": "classification"},
            "clinicaltrials":   {"target": "study_status", "type": "classification"},
            "moma":             {"target": "department", "type": "classification"}
        }
        return tasks[self.dataset_name]

    def _initialize_formatter(self) -> BaseFormatter:
        if self.target_model_type in ["diffusion", "gan"]:
            return DiffusionFormatter()
        elif self.target_model_type == "llm":
            return LLMFormatter()
        elif self.target_model_type == "tabdlm":
            return TabDLMFormatter()
        elif self.target_model_type == "tabby":
            return TabbyFormatter()

    
    
if __name__ == "__main__":
    # this is here for debugging 
    loader = UnifiedDataLoader(dataset_name="moma", target_model_type="llm")
    print(loader.raw_train.shape, loader.raw_train.head(), loader.raw_train['Title'].unique().shape)
    