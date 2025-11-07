for specialty in internal_medicine pediatrics neurology
do
    for split in 0 1 2
    do
        python3 src/mediq/mediq_baseline.py \
            --predictor_model_id meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
            --save_dir ./results/mediq/baseline/${specialty}/llama3.1-8b/ \
            --split ${split} \
            --specialty ${specialty}
    done
done