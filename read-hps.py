import os
import glob
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import sys

def view_all_best_results(directory="."):
    # Find both .log and .db files
    log_files = glob.glob(os.path.join(directory, "*.log"))
    db_files = glob.glob(os.path.join(directory, "*.db"))
    
    # Combine and sort them alphabetically
    all_files = sorted(log_files + db_files)
    
    if not all_files:
        print("No .log or .db files found in the specified directory.")
        return

    for filepath in all_files:
        filename = os.path.basename(filepath)
        
        # Split the filename and extension (e.g., 'hpo_tabby_cern', '.log')
        study_name, ext = os.path.splitext(filename)
        
        try:
            # Dynamically set the correct storage backend
            if ext == '.log':
                storage = JournalStorage(JournalFileStorage(filepath))
            elif ext == '.db':
                # SQLite needs a URL format. We use absolute path to be safe.
                abs_path = os.path.abspath(filepath)
                storage = f"sqlite:///{abs_path}"
            else:
                continue
            
            # Load the study
            study = optuna.load_study(study_name=study_name, storage=storage)
            
            # Check if the study has completed any trials successfully
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            print(f"=== {filename} ===")
            if completed_trials:
                print(f"  Best Score:  {study.best_value:.4f}")
                print(f"  Best Params: {study.best_params}")
            else:
                print("  Status: No successful trials completed yet.")
            print("-" * 50)
            
        except KeyError:
             print(f"=== {filename} ===")
             print("  Status: File exists but study not initialized or empty.")
             print("-" * 50)
        except Exception as e:
            print(f"=== {filename} ===")
            print(f"  Error loading file: {e}")
            print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[-1]
    else:
        directory = '.'
    view_all_best_results(directory)
    