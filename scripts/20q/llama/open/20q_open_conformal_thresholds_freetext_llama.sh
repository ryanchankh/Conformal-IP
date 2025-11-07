for seed in 0 1 2 3 4
do
    for n_cal_samples in 100
    do
    python3 src/20q/20q_conformal_thresholds_open_freetext.py \
        --predictor_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
        --n_iterations 21 \
        --n_cal_samples ${n_cal_samples} \
        --seed ${seed} \
        --saved_history_paths './results/20q/llama-3.1-8b/direct_freetext/n_iterations21-n_ent_samples4-seed*/' \
        --save_dir ./thresholds/20q/llama-3.1-8b/open_freetext/
    done
done
