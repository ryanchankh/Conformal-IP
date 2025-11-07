for seed in 0 1 2 3 4
do
    for n_ent_samples in 4
    do
        for alpha in 0.15
        do
            python3 src/20q/20q_conformal_open_binary.py \
                --predictor_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
                --querier_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
                --query_answerer_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
                --seed ${seed} \
                --n_iterations 21 \
                --n_queries_per_step 5 \
                --n_ent_samples ${n_ent_samples} \
                --alpha ${alpha} \
                --threshold_path ./thresholds/20q/llama-3.1-8b/open_binary/seed${seed}-n_iterations21-n_cal_samples100.json \
                --save_dir ./results/20q/llama-3.1-8b/conformal_open_binary/
        done
    done
done