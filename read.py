import os
import glob
import argparse
import optuna
from optuna.storages import JournalStorage, JournalFileStorage

def view_all_best_results(directory=".", verbose=False):
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
                print(f"  Trials completed: {len(completed_trials)}")
                print(f"  Best Score:  {study.best_value:.4f}")
                print(f"  Best Params: {study.best_params}")
            else:
                print("  Status: No successful trials completed yet.")
                
            # If verbose flag is set, print info on every trial
            if verbose:
                print("\n  --- Trial Details ---")
                for t in study.trials:
                    # t.state.name provides 'COMPLETE', 'FAIL', 'RUNNING', etc.
                    state_str = t.state.name
                    # Format the score if it exists, otherwise show N/A
                    score_str = f"{t.value:.4f}" if t.value is not None else "N/A"
                    
                    print(f"  Trial {t.number}: State={state_str:<9} Score={score_str:<10} Params={t.params}")
            
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
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="View Optuna study results.")
    parser.add_argument("-d", "--dir", type=str, default=".", 
                        help="Directory containing the .log and .db files (default: current directory).")
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Print detailed information for every trial run.")
    args = parser.parse_args()

    # Pass the arguments to the main function
    view_all_best_results(directory=args.dir, verbose=args.verbose)

    