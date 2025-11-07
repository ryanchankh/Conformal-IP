for seed in {0..4}
do
    for n_cal_samples in 100
    do
        python3 src/20q/20q_conformal_thresholds_closed_freetext.py \
            --predictor_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
            --n_iterations 21 \
            --n_cal_samples ${n_cal_samples} \
            --seed ${seed} \
            --qry_ans_path ./results/20q/llama-3.1-8b/query_answers_closed/n_answers_per_query10/ \
            --save_dir ./thresholds/20q/llama-3.1-8b/closed_freetext/
    done
done
