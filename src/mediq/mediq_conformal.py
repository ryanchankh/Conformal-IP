import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
from typing import List
import json
import os
from itertools import product
from tqdm import tqdm

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
    set_seed
)
from datasets import Dataset
from mediq_dataset import MediQDataset
from utils import (
    save_run,
    read_json,
    extract_and_parse_dict,
    compute_entropy_formula,
    compute_upper_bound_formula,
    format_predictor_messages,
    format_querier_messages,
    format_query_answerer_messages,
)
from prompts import (
    QUERY_ANSWERER_SYSTEM_PROMPT,
    QUERY_ANSWERER_PROMPT,
    PREDICTOR_SYSTEM_PROMPT,
    PREDICTOR_PROMPT,
    QUERIER_SYSTEM_PROMPT,
    QUERIER_PROMPT,
)

class MediQ:
    def __init__(
        self,
        predictor_model_id: str,
        querier_model_id: str,
        query_answerer_model_id: str,
        n_ent_samples: int = 64,
        n_iterations: int = 20,
        n_queries_per_step: int = 10,
        train_dataset: Dataset = None,
        thresholds: dict = None,
        alpha: float = 0.05,
        specialty: str = None,
        seed: int = 0,
    ):
        self.predictor_model_id = predictor_model_id
        self.querier_model_id = querier_model_id
        self.query_answerer_model_id = query_answerer_model_id
        self.n_ent_samples = n_ent_samples
        self.n_iterations = n_iterations
        self.n_queries_per_step = n_queries_per_step
        self.train_dataset = train_dataset
        self.thresholds = thresholds
        self.alpha = alpha
        self.specialty = specialty
        self.seed = seed

        set_seed(self.seed)
        self.setup_models()

    def setup_models(self):        
        # Posterior model
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.predictor_tokenizer = AutoTokenizer.from_pretrained(
            self.predictor_model_id,
            trust_remote_code=True,
            torch_dtype='auto',
            attn_implementation="flash_attention_2",
            quantization_config=quantization_config,
            device_map="auto",
        )
        self.predictor_model = AutoModelForCausalLM.from_pretrained(
            self.predictor_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.predictor_tokenizer.pad_token = self.predictor_tokenizer.eos_token
        self.predictor_pipeline = pipeline(
            'text-generation',
            model=self.predictor_model,
            tokenizer=self.predictor_tokenizer
        )

    def get_response(self, messages):
        outputs = self.predictor_pipeline(messages, do_sample=True,
            temperature=0.7, max_new_tokens=500)
        return outputs[0]['generated_text'][-1]['content']

    def get_next_query(self, history, queries):
        sampled_queries = queries

        # estimate entropy of each query
        entropy_estimates, entropy_estimates_emp = self.estimate_entropy(history, sampled_queries)
        print('SAMPLED QUERIES' + '='*23)
        [print('\t', qry, est, est_emp) for qry, est, est_emp in zip(sampled_queries,
            entropy_estimates, entropy_estimates_emp)]
        print('MIN EQUAL:', entropy_estimates.argmin() == entropy_estimates_emp.argmin())
        
        # select query
        min_ent = entropy_estimates.min()
        min_ent_idx = entropy_estimates.argmin()
        min_ent_query = sampled_queries[min_ent_idx]
        sampled_queries.pop(min_ent_idx)
        return min_ent_query, min_ent.item(), sampled_queries, \
            entropy_estimates.tolist(), entropy_estimates_emp.tolist()

    def sample_data(self, n_ent_samples):
        sample_indices = np.random.choice(len(self.train_dataset), n_ent_samples, replace=True)
        questions, options, facts, init_infos, labels = [], [], [], [], []
        for sample_idx in sample_indices:
            sample_dict = self.train_dataset[sample_idx.item()]
            questions.append(sample_dict['question'])
            options.append(eval(sample_dict['options']))
            facts.append(eval(sample_dict['facts']))
            init_infos.append(eval(sample_dict['context'])[0])
            labels.append({"A": 0, "B": 1, "C": 2, "D": 3}[sample_dict['answer_idx']])
        return questions, options, facts, init_infos, labels
            
        
    def estimate_entropy(self, history, sampled_queries):
        """Estimate entropy of each query."""
        
        # sample data
        sampled_questions, sampled_options, sampled_facts, sampled_infos, sampled_labels \
            = self.sample_data(self.n_ent_samples)
        query_answers = self.get_query_answers(sampled_facts, sampled_queries)

        entropies_bound = torch.ones(len(sampled_queries)) * torch.inf
        entropies_emp = torch.ones(len(sampled_queries)) * torch.inf
        for qry_j, query in enumerate(sampled_queries):
            # q(X) U history
            history_with_qx = [history.copy() + [(query, ans.item())] for ans in query_answers[:, qry_j]]
            
            # compute posterior P(Y | q(X) U history))
            posteriors_j, _ = self.compute_posteriors(history_with_qx, sampled_questions,
                sampled_options, sampled_infos, max_new_tokens=1)
            
            # entropy emp
            entropy_j_emp = compute_entropy_formula(posteriors_j)

            # entropy bound
            length_k = len(history_with_qx[0])
            qhat = self.thresholds[str(length_k)][str(self.alpha)]
            entropy_j_bound = compute_upper_bound_formula(posteriors_j, sampled_labels,
                self.alpha, qhat, label_sizes=4)

            # store results
            entropies_bound[qry_j] = entropy_j_bound
            entropies_emp[qry_j] = entropy_j_emp
        return entropies_bound, entropies_emp

    def get_query_answer(self, facts, query):
        messages = format_query_answerer_messages(facts, query)
        response = self.get_response(messages)
        response = response.replace('\n', '')
        return response

    def get_query_answers(self, facts_lst, queries):
        assert isinstance(facts_lst, List) or isinstance(facts_lst, np.ndarray), 'samples need to be of type List'
        assert isinstance(queries, List) or isinstance(queries, np.ndarray), 'queries need ot be of type List'
  
        query_answers = []
        for fact_i, facts in enumerate(facts_lst):
            query_answers_for_label = []
            for query in tqdm(queries, desc=f'get_query_answers {fact_i}: queries'):
                qry_ans = self.get_query_answer(facts, query)
                query_answers_for_label.append(qry_ans)
            query_answers.append(query_answers_for_label)
        return np.array(query_answers)

    def compute_posteriors(self, histories, questions, options, init_infos, max_new_tokens=1):
        """Compute posterior from classifier."""
        posteriors, responses = [], []
        iterator = tqdm(
            zip(histories, questions, options, init_infos),
            total=len(histories),
            desc='computing posteriors'
        )
        for history, question, option, init_info in iterator:
            messages = format_predictor_messages(history, question, option, init_info)
            inputs_str = self.predictor_tokenizer.apply_chat_template([messages],
                add_generation_prompt=False, tokenize=False, continue_chat_template=True)
            if self.predictor_model_id == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
                inputs_str = [inp_str[:-10] for inp_str in inputs_str] # get around bug where continue_chat_template adds extra tokens
            else:
                raise ValueError(f"Model {self.predictor_model_id} not supported.")
            inputs_token = self.predictor_tokenizer(inputs_str,
                return_tensors="pt", padding=True).to('cuda')
            outputs = self.predictor_model.generate(**inputs_token,
                max_new_tokens=max_new_tokens, output_logits=True, return_dict_in_generate=True,
                pad_token_id=self.predictor_tokenizer.eos_token_id)
            batch_posteriors = self.get_class_probs_from_logits(outputs['logits'][0].cpu())
            response = self.predictor_tokenizer.batch_decode(outputs.sequences)[0]
            posteriors.append(batch_posteriors)
            responses.append(response)
        return torch.vstack(posteriors), responses

    def get_class_probs_from_logits(self, logits):
        if self.predictor_model_id == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
            class_indices = torch.tensor([32, 33, 34, 35], dtype=torch.long)
        else:
            raise ValueError(f"Model {self.predictor_model_id} not supported.")
        return torch.softmax(logits[:, class_indices], dim=-1)

    def run_one_sample(self, sample_dict):
        # parse sample
        question = sample_dict['question']
        option = eval(sample_dict['options'])
        facts = eval(sample_dict['facts'])
        init_info = eval(sample_dict['context'])[0]
        queries = sample_dict['queries']
        n_iterations = min(len(queries), self.n_iterations) + 1

        predictions = []
        history = []
        sampled_queries = []
        entropy_selected_query = []
        entropy_estimates = []
        entropy_estimates_emp = []
        posteriors = []
        for k in range(n_iterations):
            print(f"ITERATION: {k}" + '='*23, flush=True)
            # predict
            posterior_k, predict_k = self.compute_posteriors([history], [question],
                [option], [init_info], max_new_tokens=400)
            print(f"PREDICT:" + '='*23 + f"\n{predict_k[0]}", flush=True)
            predictions.append(predict_k[0])
            
            if k == (n_iterations - 1):
                break
            
            # query
            query_k, entropy_selected_query_k, sampled_queries_k, entropy_estimates_k, \
                entropy_estimates_emp_k = self.get_next_query(history, queries)
            print(f"QUERY:" + "="*23 + f"\n{query_k.strip()}", flush=True)

            # answer
            answer_k = self.get_query_answer(facts, query_k)
            print(f"ANSWER:" + "="*23 + f"\n{answer_k.strip()}", flush=True) 
            
            history.append((query_k, answer_k))
            sampled_queries.append(sampled_queries_k)
            entropy_selected_query.append(entropy_selected_query_k)
            entropy_estimates.append(entropy_estimates_k)
            entropy_estimates_emp.append(entropy_estimates_emp_k)
            posteriors.append(posterior_k.tolist())

        results_dict = {
            'predictions': predictions,
            'history': history,
            'sampled_queries': sampled_queries,
            'entropy_selected_query': entropy_selected_query,
            'entropy_estimates': entropy_estimates,
            'entropy_estimates_emp': entropy_estimates_emp,
            'posteriors': posteriors,
        }
        return results_dict
    
def main(args):
    
    # save dir 
    save_dir = f"{args.save_dir}/n_iterations{args.n_iterations}-n_queries_per_step{args.n_queries_per_step}-n_ent_samples{args.n_ent_samples}-alpha{args.alpha}/split{args.split}/"
    os.makedirs(save_dir, exist_ok=True)

    # load datsts            
    mediqa_dataset = MediQDataset('./data/mediQ/', args.specialty, args.split)
    train_dataset, test_dataset, _ = mediqa_dataset.load()
    thresholds = read_json(args.thresholds_path)

    # play game
    game = MediQ(
        predictor_model_id=args.predictor_model_id,
        querier_model_id=args.querier_model_id,
        query_answerer_model_id=args.query_answerer_model_id,
        n_iterations=args.n_iterations,
        n_ent_samples=args.n_ent_samples,
        n_queries_per_step=args.n_queries_per_step,
        train_dataset=train_dataset,
        specialty=args.specialty,
        alpha=args.alpha,
        thresholds=thresholds
    )
    iterator = tqdm(enumerate(test_dataset), total=len(test_dataset), desc='test samples')
    for sample_i, test_sample_dict in iterator:
        save_path = os.path.join(save_dir, f'test_sample{sample_i}.json')
        
        # skip if path exists
        if os.path.exists(save_path):
            print(f"Already done for {sample_i}. Skipping.")
            continue
        
        # run one sample
        results = game.run_one_sample(test_sample_dict)
        results['sample_dict'] = test_sample_dict
        results['params'] = vars(args)
        save_run(save_path, results)

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_iterations', type=int, default=20)
    parser.add_argument('--n_ent_samples', type=int, default=64)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--specialty', type=str, required=True)
    parser.add_argument('--n_queries_per_step', type=int, default=1)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--thresholds_path', type=str, required=True)
    parser.add_argument('--predictor_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--querier_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--query_answerer_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--save_dir', type=str, default='./results/thresholds/20q/llama-3.1-8b/')
    return parser.parse_args()

if __name__ == '__main__':
    args = parseargs()
    main(args)
        