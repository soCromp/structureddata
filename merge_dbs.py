import optuna
import glob

# This is the single database you will point the dashboard at
master_storage = "sqlite:///merged.db"

# Grab all the individual dataset databases in the folder
db_files = glob.glob("*.db")
db_files = [f for f in db_files if f != "merged.db"] # prevent infinite loop

for db_file in db_files:
    source_storage = f"sqlite:///{db_file}"
    
    # Check what studies (e.g., hpo_tabby_olist, hpo_icl_bayesian) are in this file
    try:
        studies = optuna.get_all_study_summaries(storage=source_storage)
    except Exception as e:
        print(f"Skipping {db_file}: {e}")
        continue
        
    for study in studies:
        print(f"Copying '{study.study_name}' from {db_file} -> merged.db")
        try:
            optuna.copy_study(
                from_study_name=study.study_name,
                from_storage=source_storage,
                to_storage=master_storage,
                to_study_name=study.study_name
            )
        except optuna.exceptions.DuplicatedStudyError:
            print(f"  - Study '{study.study_name}' already exists in master. Skipping.")

print("\nMerge complete! You can now launch the dashboard:")
print("optuna-dashboard sqlite:///merged.db --port 8080")
