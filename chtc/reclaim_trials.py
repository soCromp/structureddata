import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optuna.trial import TrialState
import os

# Load your Journal log
for file in os.listdir('/staging/c/cromp/sd/hps'): 
    if not file.endswith('.log'):
        continue
    print('     ', file)
    storage = JournalStorage(JournalFileStorage(os.path.join('/staging/c/cromp/sd/hps', file)))
    study = optuna.load_study(study_name=file[:-4], storage=storage)

    # Hunt down the zombie trials and any trials that legitimately failed
    stuck_trials = [t for t in study.trials if t.state in [TrialState.RUNNING, TrialState.FAIL]]

    for trial in stuck_trials:
        print(f"     Re-enqueuing crashed trial {trial.number} with params: {trial.params}")
        # Push the exact parameters to the front of the line
        study.enqueue_trial(trial.params)
        
    print('RECOVERED', len(stuck_trials), 'TRIALS:', file[:-4])
    