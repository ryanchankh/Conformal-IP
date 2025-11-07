for seed in 0 1 2 3 4
do
    for n_ent_samples in 4
    do
        for alpha in 0.1
        do
            python3 src/20q/20q_conformal_closed_binary.py \
                --predictor_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
                --querier_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
                --seed ${seed} \
                --n_iterations 21 \
                --n_ent_samples ${n_ent_samples} \
                --alpha ${alpha} \
                --threshold_path ./thresholds/20q/llama-3.1-8b/closed_binary/seed${seed}-n_iterations21-n_cal_samples100.json \
                --qry_ans_path ./data/Animals_with_Attributes2/query_bank_closed_yesno/dicts/ \
                --save_dir ./results/20q/llama-3.1-8b/conformal_closed_binary/
        done
    done
done
