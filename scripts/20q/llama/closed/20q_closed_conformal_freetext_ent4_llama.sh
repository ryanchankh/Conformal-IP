for seed in 0 1 2 3 4
do
    for n_ent_samples in 8
    do
        for alpha in 0.1
        do
            python3 src/20q/20q_conformal_closed_freetext.py \
                --predictor_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
                --querier_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
                --seed ${seed} \
                --n_iterations 21 \
                --n_ent_samples ${n_ent_samples} \
                --alpha ${alpha} \
                --threshold_path ./thresholds/20q/llama-3.1-8b/closed_freetext/seed${seed}-n_iterations21-n_cal_samples100.json \
                --qry_ans_path ./results/20q/llama-3.1-8b/query_answers_closed/n_answers_per_query10/ \
                --save_dir ./results/20q/llama-3.1-8b/conformal_closed_freetext/
        done
    done
done
