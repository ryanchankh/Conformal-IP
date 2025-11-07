# for seed in 0 1 2 3 4
# for seed in 10 11 12 13 14
# for seed in 20 21 22 23 24
# for seed in 30 31 32 33 34
for seed in 40 41 42 43 44
do
    for n_ent_samples in 4
    do
        python3 src/20q/20q_direct_open_freetext.py \
            --predictor_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
            --querier_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
            --query_answerer_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
            --seed ${seed} \
            --n_iterations 21 \
            --n_queries_per_step 1 \
            --n_ent_samples ${n_ent_samples} \
            --save_dir ./results/20q/llama-3.1-8b/direct_freetext/
    done
done
