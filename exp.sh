# !/bin/bash
# run from structureddata dir

Datasets=("moma" "lob")
conda activate sd

# prep all datasets (since not all model envs have the data construction libraries installed)
for dataset in $Datasets; do:
    python handler.py $dataset

# tabdiff dataset prep
cd models/TabDiff
conda activate tabdiff 
for dataset in $Datasets; do:
    python process_dataset.py --dataname $dataset
done

cd ..

for dataset in $Datasets; do:
    # ctganp (runs all 3 trials)
    cd CTAB-GAN-Plus
    conda activate ctganp 
    python run_ctganp.py $dataset

    for i in {1..3}; do
        cd ../tabby
        conda activate tabby
        python trainplain.py -t -p /mnt/data/sonia/sd/tabby/$(dataset)/$(i) -d $dataset -mh
        python trainplain.py -p /mnt/data/sonia/sd/tabby/$(dataset)/$(i) -d $dataset -mh -n 10000
        cp /mnt/data/sonia/sd/tabby/$(dataset)/$(i)/samples.csv ../../synth/$dataset/tabby_$(i).csv

        cd ../TabDiff
        conda activate tabdiff
        python run_tabdiff.py --dataset $dataset --mode train
        python main.py --dataname $dataset --mode test --report --no_wandb
    done
done
