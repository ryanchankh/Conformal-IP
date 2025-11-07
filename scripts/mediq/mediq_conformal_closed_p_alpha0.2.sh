for split_idx in 0 1 2
do
    # for specialty in internal_medicine pediatrics neurology
    for specialty in pediatrics
    do
        python3 src/mediq/mediq_conformal.py \
            --predictor_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
            --querier_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
            --query_answerer_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
            --seed 0 \
            --split ${split_idx} \
            --specialty ${specialty} \
            --n_iterations 10 \
            --n_queries_per_step 5 \
            --n_ent_samples 12 \
            --alpha 0.2 \
            --thresholds_path ./thresholds/mediq/llama-3.1-8b/${specialty}/closed_split${split_idx}/seed0-n_iterations21-n_cal_samples200.json \
            --save_dir ./results/mediq/conformal_closed/${specialty}/llama3.1-8b/
    done
done
