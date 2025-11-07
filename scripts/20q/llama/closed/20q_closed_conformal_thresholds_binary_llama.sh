for seed in {0..4}
do
    for n_cal_samples in 100
    do
        python3 src/20q/20q_conformal_thresholds_closed_binary.py \
            --predictor_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
            --n_iterations 21 \
            --n_cal_samples ${n_cal_samples} \
            --seed ${seed} \
            --qry_ans_path ./data/Animals_with_Attributes2/query_bank_closed_yesno/dicts/ \
            --save_dir ./thresholds/20q/llama-3.1-8b/closed_binary/
    done
done
