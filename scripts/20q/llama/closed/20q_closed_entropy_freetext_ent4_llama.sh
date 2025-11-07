for seed in {0..4}
do
    for n_ent_samples in 4
    do
        python3 src/20q/20q_entropy_closed_freetext.py \
            --predictor_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
            --seed ${seed} \
            --n_iterations 21 \
            --n_ent_samples ${n_ent_samples} \
            --qry_ans_path ./results/20q/llama-3.1-8b/query_answers_closed/n_answers_per_query10 \
            --save_dir ./results/20q/llama-3.1-8b/entropy_closed_freetext/
    done
done