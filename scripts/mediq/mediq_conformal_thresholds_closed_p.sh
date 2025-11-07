seed=0
for split_idx in 0 1 2
do
    for specialty in pediatrics
    do
        for n_cal_samples in 200
        do
            python3 src/mediq/mediq_conformal_thresholds.py \
                --predictor_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
                --query_answerer_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
                --n_iterations 21 \
                --n_cal_samples ${n_cal_samples} \
                --alphas 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 \
                --specialty ${specialty} \
                --split_idx ${split_idx} \
                --seed ${seed} \
                --save_dir ./thresholds/mediq/llama-3.1-8b/${specialty}/closed_answer_split${split_idx}/
        done
    done
done

