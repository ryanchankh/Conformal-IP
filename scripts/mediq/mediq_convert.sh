for split_idx in 0 1 2
do
    for specialty in internal_medicine pediatrics neurology
    # for specialty in neurology
    do
        python3 src/mediq/mediq_convert.py \
            --predictor_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
            --seed 0 \
            --split ${split_idx} \
            --specialty ${specialty} \
            --save_dir ./results/mediq/convert/${specialty}/llama3.1-8b/
    done
done



