#!/bin/bash
# run from structureddata dir

Datasets=("honeypot" "stroke" "cern" "lob" "moma" "olist" "bayesian")
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
    cd CTAB-GAN-Plus
    conda activate ctganp 
    python run_ctganp.py 1 "$dataset"

    cd ../ICL
    conda activate tabby
    python run_icl.py --dataset "$dataset" --model_id /mnt/data/zoo/meta-llama/Meta-Llama-3-8B --num_samples 30 # only do one trial for ICL

    cd ../tabby
    conda activate tabby
    python trainplain.py -t -p /mnt/data/sonia/sd/tabby/${dataset}/debug -d "$dataset" -mh -steps 5000 -n 0 -l1 --local -eff
    python trainplain.py -p /mnt/data/sonia/sd/tabby/${dataset}/debug -d "$dataset" -mh -n 50 --local -l1 -eff
    cp /mnt/data/sonia/sd/tabby/${dataset}/debug/samplesclean.csv ../../synth/"$dataset"/tabby_debug.csv

    cd ../TabDiff
    conda activate tabdiff
    python main.py --dataname $dataset --mode train  
    python main.py --dataname $dataset --mode test --report --no_wandb 
    cp synthetic/${dataset}/test.csv ../../synth/${dataset}/tabdiff_debug.csv 

    cd ../TabDLM
    conda activate tabdlm
    PYTHONPATH=. python main.py train --dataset_name $dataset --description "_tabdlm" \
                --epochs 1 --batch_size 1 --batch_accum 128 --lora_r 4 --lora_alpha 128 \
                --answer_len 160 --loss_type no_divide_pmask --bf16 
    PYTHONPATH=. python main.py sample --dataset_name $dataset --description "_tabdlm" \
                --save_description "_tabdlm_synth" --do_sampling --bf16 --use_best_ckp \
                --gen_length 160 --block_length 160 --sample_step 160 --temperature 1.0 \
                --sample_batch_size 8 --seed 1

    cd ../TabKG
    conda activate tabkg # need to also run VLLM
    python main.py --method crkg --data $dataset --ensemble "gpt,gpt,gpt,gpt,gpt" --temp_range "0.1,0.2,0.3,0.4,0.5"
    cp results/${dataset}/CRKG_FilteredOutput.csv ../../synth/"$dataset"/tabkg_debug.csv

    cd ..

done
