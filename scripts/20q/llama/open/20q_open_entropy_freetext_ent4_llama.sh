for seed in 0 1 2 3 4
do
    for n_ent_samples in 4
    do
        python3 src/20q/20q_entropy_open_freetext.py \
            --predictor_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
            --querier_model_id meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
            --query_answerer_model_id meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo \
            --seed ${seed} \
            --n_iterations 21 \
            --n_queries_per_step 5 \
            --n_ent_samples ${n_ent_samples} \
            --save_dir ./results/20q/llama-3.1-8b/entropy_freetext/
    done
done

