#!/bin/bash
# run from structureddata dir

Datasets=("lob" "moma")
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sd

# prep all datasets (since not all model envs have the data construction libraries installed)
for dataset in "${Datasets[@]}"; do
    python handler.py "$dataset"
done

# tabdiff dataset prep
cd models/TabDiff
conda activate tabdiff 
for dataset in "${Datasets[@]}"; do
    python process_dataset.py --dataname "$dataset"
done

cd ..
for dataset in "${Datasets[@]}"; do
    # ctganp (runs all 3 trials)
    cd CTAB-GAN-Plus
    conda activate ctganp 
    python run_ctganp.py 3 "$dataset"

    cd ../ICL
    conda activate tabby
    python run_icl.py --dataset "$dataset" --model_id /mnt/data/zoo/meta-llama/Meta-Llama-3-8B # only do one trial for ICL

    for i in {1..3}; do
        cd ../tabby
        conda activate tabby
        
        python trainplain.py -t -p /mnt/data/sonia/sd/tabby/${dataset}/$i -d "$dataset" -mh -e 5 -n 0 -l1 --local  -eff
        python trainplain.py -p /mnt/data/sonia/sd/tabby/${dataset}/$i -d "$dataset" -mh -n 50 --local -l1 -eff
        cp /mnt/data/sonia/sd/tabby/${dataset}/$i/samplesclean.csv ../../synth/"$dataset"/tabby_$i.csv

        cd ../TabDiff
        conda activate tabdiff
        python main.py --dataname $dataset --mode train
        python main.py --dataname $dataset --mode test --report --no_wandb
        cp synthetic/${dataset}/test.csv ../../synth/${dataset}/tabdiff_$i.csv

        cd ../TabDLM
        conda activate tabdlm
        TOKENIZERS_PARALLELISM=false PYTHONPATH=. python main.py train --dataset_name $dataset \
                    --description "_tabdlm" --epochs 10 --batch_size 32 --batch_accum 32 \
                    --loss_type no_divide_pmask  --lora_r 16 --lora_alpha 64 --bf16
        TOKENIZERS_PARALLELISM=false PYTHONPATH=. python main.py sample --dataset_name $dataset \
                    --description "_tabdlm" --save_description "_tabdlm_synth" --do_sampling \
                    --use_best_ckp --temperature 1.0 --sample_batch_size 32 --seed $i --proportion 200
        cp result/${dataset}/synthetic_result/_tabdlm_tabdlm_synth.csv ../../synth/${dataset}/tabdlm_$i.csv

        cd ../TabKG
        conda activate tabkg # need to also run VLLM
        python main.py --method crkg --data $dataset --ensemble "gpt,gpt,gpt,gpt,gpt" --temp_range "0.1,0.2,0.3,0.4,0.5"
        cp results/${dataset}/CRKG_FilteredOutput.csv ../../synth/"$dataset"/tabkg_$i.csv

        cd ..
    done
done
