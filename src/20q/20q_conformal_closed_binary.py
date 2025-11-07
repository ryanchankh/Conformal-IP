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
    BitsAndBytesConfig
)

from utils import (
    read_class_names,
    add_to_history,
    add_to_prompt,
    read_txt,
    read_attr,
    save_json,
    read_json,
    save_run,
    extract_and_parse_dict

)
from prompts import (
    QUERY_ANSWERER_SYSTEM_PROMPT,
    PREDICTOR_SYSTEM_PROMPT,
    PREDICTOR_PROMPT,
    QUERIER_SYSTEM_PROMPT,
    QUERIER_PROMPT
    
)

from together import Together


class TwentyQuestions:
    def __init__(
        self,
        predictor_model_id: str,
        querier_model_id: str,
        n_ent_samples: int = 64,
        n_iterations: int = 20,
        class_names: List[str] = None,
        thresholds = None,
        alpha: float = 0.01,
        query_bank: List[str] = None,
        query_bank_answers = None,
        seed: int = 0,
    ):
        self.predictor_model_id = predictor_model_id
        self.querier_model_id = querier_model_id
        self.n_ent_samples = n_ent_samples
        self.n_iterations = n_iterations
        self.class_names = class_names
        self.thresholds = thresholds
        self.alpha = alpha
        self.query_bank = query_bank
        self.query_bank_answers = query_bank_answers
        self.seed = seed

        self.class_names_str = ', '.join([f"'{name.replace('+', ' ')}'" for name in class_names])
        self.n_classes = len(class_names)

        self.setup_models()

    def setup_models(self):
        # APIs
        self.client = Together()
        
        # Posterior model
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.predictor_tokenizer = AutoTokenizer.from_pretrained(
            self.predictor_model_id,
            trust_remote_code=True,
            torch_dtype='auto',
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.predictor_model = AutoModelForCausalLM.from_pretrained(
            self.predictor_model_id,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype='auto',
        )
        self.predictor_tokenizer.pad_token = self.predictor_tokenizer.eos_token
        self.predictor_pipeline = pipeline(
            'text-generation',
            model=self.predictor_model,
            tokenizer=self.predictor_tokenizer
        )

    def get_response(self, model_id, messages):
        outputs = self.predictor_pipeline(messages, temperature=0.7, max_new_tokens=300)
        return outputs[0]['generated_text'][-1]['content']

    def get_next_query(self, history):
        sampled_queries = self.sample_queries(history)

        # estimate entropy of each query
        entropy_estimates, entropy_estimates_emp = self.estimate_entropy(history, sampled_queries)
        [print('\t', qry, est) for qry, est in zip(sampled_queries, entropy_estimates)]
        
        # select query
        min_ent = entropy_estimates.min()
        min_ent_idx = entropy_estimates.argmin()
        min_ent_query = sampled_queries[min_ent_idx]
        return min_ent_query, min_ent.item(), sampled_queries, entropy_estimates.tolist(), entropy_estimates_emp[min_ent_idx].item()

    def sample_data(self, n_samples):
        """Sampling data for AwA2 dataset is simply getting a set of labels. """
        sample_indices = np.random.choice(self.n_classes, n_samples, replace=True)
        return self.class_names[sample_indices], sample_indices

    def sample_queries(self, history): #TODO
        """Sample queries from the query bank."""
        queries = self.query_bank
        # remove already asked queries
        existing_queries = [q for q, _ in history]
        print(f"EXISTING QUERIES: {existing_queries}")
        queries = [q for q in queries if q not in existing_queries]
        return queries

    def estimate_entropy(self, history, sampled_queries):
        sampled_labels, sampled_label_idxs = self.sample_data(self.n_ent_samples)
        query_answers = self.get_query_answers(sampled_labels, sampled_queries)

        entropies_bound = torch.ones(len(sampled_queries)) * torch.inf
        entropies_emp = torch.ones(len(sampled_queries)) * torch.inf
        for qry_j, query in enumerate(sampled_queries):
            start = time.time()
           
            # add q(X) to history 
            history_with_qx = [history.copy() + [(query, ans.item())] for ans in query_answers[:, qry_j]]
            
            # compute posterior
            posteriors_j, _ = self.compute_posteriors(history_with_qx)

            # compute posterior
            length_k = len(history_with_qx[0])
            qhat = self.thresholds[str(length_k)][str(self.alpha)]
            
            # compute entropy empirical
            entropy_j_emp = self.compute_entropy_formula(posteriors_j)
            
            # compute bound
            psets = self.compute_predict_sets(posteriors_j, qhat)
            include_inds = torch.where(psets[torch.arange(self.n_ent_samples), sampled_label_idxs]==1, True, False)
            include_psets_sizes = psets[include_inds].sum(1)
            exclude_psets_sizes = psets[~include_inds].sum(1)
            entropy_j_bound = self.compute_upper_bound_formula(include_psets_sizes, exclude_psets_sizes, self.n_classes, self.alpha)

            # store results
            entropies_bound[qry_j] = entropy_j_bound
            entropies_emp[qry_j] = entropy_j_emp
        return entropies_bound, entropies_emp

    def get_query_answer(self, label, query):
        possible_answers = self.query_bank_answers[label][query]
        answer = np.random.choice(possible_answers, None)
        return answer

    def get_query_answers(self, labels, queries):
        assert isinstance(labels, List) or isinstance(labels, np.ndarray), 'samples need to be of type List'
        assert isinstance(queries, List) or isinstance(queries, np.ndarray), 'queries need ot be of type List'
  
        query_answers = []
        for label in labels:
            query_answers_for_label = []
            for query in queries:
                qry_ans = self.get_query_answer(label, query)
                query_answers_for_label.append(qry_ans)
            query_answers.append(query_answers_for_label)
        return np.array(query_answers)

    def compute_entropy_formula(self, test_posteriors):
        probs = test_posteriors
        assert torch.all(probs >= 0.)
        probs = torch.clamp(probs, min=1e-5)
        return -(probs * torch.log2(probs)).sum(-1).mean()

    def compute_upper_bound_formula(self, include_pset_sizes, exclude_pset_sizes, label_sizes, alpha):
        n = include_pset_sizes.shape[0] + exclude_pset_sizes.shape[0]
        h_b = -alpha * np.log2(alpha) - (1-alpha) * np.log2(1-alpha)
        alpha_n = alpha - (1 / (n + 1))
        if len(include_pset_sizes) == 0:
            include_exp = torch.tensor([0.]).cuda()
        else:
            include_exp = (1 - alpha_n) * torch.log2(include_pset_sizes).mean()
        if len(exclude_pset_sizes) == 0:
            exclude_exp = torch.tensor([0.]).cuda()
        else:
            exclude_exp = alpha * torch.log2(label_sizes - exclude_pset_sizes).mean()
        final_bound = h_b + include_exp + exclude_exp
        return final_bound

    def compute_predict_sets(self, test_probs, qhat):
        """Compute prediction set."""
        # 3: form prediction sets
        prediction_sets = torch.where(test_probs >= (-qhat), 1., 0)
        return prediction_sets.float()

    def compute_posteriors(self, histories, max_new_tokens=1):
        """Compute posterior from classifier."""
        posteriors, responses = [], []
        for history in tqdm(histories, total=len(histories), desc='computing posteriors'):
            messages = format_predictor_messages(self.class_names_str, history)
            inputs_str = self.predictor_tokenizer.apply_chat_template([messages], add_generation_prompt=False, tokenize=False, continue_chat_template=True)
            if self.predictor_model_id == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
                inputs_str = [inp_str[:-10] for inp_str in inputs_str] # get around bug where continue_chat_template adds extra tokens
            elif self.predictor_model_id == 'Qwen/Qwen2.5-7B-Instruct':
                inputs_str = [inp_str[:-11] for inp_str in inputs_str]
            elif self.predictor_model_id == 'microsoft/Phi-3-small-128k-instruct':
                inputs_str = [inp_str[:-len('<|end|>\n<|endoftext|>')] for inp_str in inputs_str]
            else:
                raise ValueError(f"Model {self.predictor_model_id} not supported.")
            inputs_token = self.predictor_tokenizer(inputs_str, return_tensors="pt", padding=True).to('cuda')
            outputs = self.predictor_model.generate(**inputs_token, max_new_tokens=max_new_tokens, output_logits=True, return_dict_in_generate=True, pad_token_id=self.predictor_tokenizer.eos_token_id)
            batch_posteriors = self.get_class_probs_from_logits(outputs['logits'][0].cpu())
            response = self.predictor_tokenizer.batch_decode(outputs.sequences)[0]
            posteriors.append(batch_posteriors)
            responses.append(response)
        return torch.vstack(posteriors), responses
    
    def get_class_probs_from_logits(self, logits):
        if self.predictor_model_id == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
            class_indices = torch.tensor([41389,  1167, 46840, 25534,   294, 25685, 14880, 44656, 22408,
                40132, 52835,  6437, 41926,  1099,  4502, 19923, 43627, 46298,
                99269,  4647], dtype=torch.long)
        elif self.predictor_model_id == 'Qwen/Qwen2.5-7B-Instruct':
            class_indices = torch.tensor([40289,  1147, 45740, 24518,   294, 24660, 14538, 43556, 21669,
                39032, 51735,  6303, 40826,  1081,  4403, 19362, 42527, 45198,
                98169,  4544], dtype=torch.long)
        elif self.predictor_model_id == 'microsoft/Phi-3-small-128k-instruct':
            class_indices = torch.tensor([41389,  1167, 46840, 25534,   294, 25685, 14880, 44656, 22408,
                40132, 52835,  6437, 41926,  1099,  4502, 19923, 43627, 46298, 99269,  4647], dtype=torch.long)
        else:
            raise ValueError(f"Model {self.predictor_model_id} not supported.")
        return torch.softmax(logits[:, class_indices], dim=-1)

    def run_one_sample(self, label):
        predictions = []
        history = []
        sampled_queries = []
        entropy_selected_query = []
        entropy_estimates = []
        entropy_estimates_emp = []
        posteriors = []
        for k in range(self.n_iterations):
            print(f"ITERATION: {k}", flush=True)
            # predict
            posterior_k, predict_k = self.compute_posteriors([history], max_new_tokens=100)
            print(f"PREDICT: {predict_k}", flush=True)
            predictions.append(predict_k)
            
            if k == self.n_iterations - 1:
                break
            
            # query
            query_k, entropy_selected_query_k, sampled_queries_k, entropy_estimates_k, entropy_estimates_emp_k \
                = self.get_next_query(history)
            print(f"QUERY: {query_k.strip()}", flush=True)
            
            # answer
            answer_k = self.get_query_answer(label, query_k)
            print(f"ANSWER: {answer_k.strip()}", flush=True) 
            
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
            'label': label,
            'posteriors': posteriors,
        }
        return results_dict
    
def format_predictor_messages(class_names_str, history):
    # input prompt
    if len(history) == 0:
        input_prompt = "You have not gathered any information yet. Please make a random guess."
    else:
        input_prompt = "Here is the information you have gathered.\n"
        for k, (q, qx) in enumerate(history):
            input_prompt += f"{k+1}. {q} {qx}\n"
            
    messages = [
        {"role": "system", "content": PREDICTOR_SYSTEM_PROMPT.format(class_names=class_names_str)},
        {"role": "user", "content": f"{input_prompt}\n{PREDICTOR_PROMPT.format(class_names=class_names_str)}"}
    ]

    messages += [{"role": "assistant", "content": f"The animal is:"}]
    return messages


def main(args):
    
    # save dir 
    save_path = f"{args.save_dir}/n_iterations{args.n_iterations}-n_ent_samples{args.n_ent_samples}-alpha{args.alpha}-seed{args.seed}/"
    os.makedirs(save_path, exist_ok=True)

    # class names
    class_names = read_class_names('data/Animals_with_Attributes2/classes20.txt')
    query_bank = read_txt('data/Animals_with_Attributes2/query_bank_closed/queries.txt')
    query_bank_answers = {cls_name: read_json(os.path.join(args.qry_ans_path, f'{cls_name}.json')) \
        for cls_name in class_names}
    print(class_names)    

    # conformal thresholds
    thresholds = read_json(args.threshold_path)

    # play game
    game = TwentyQuestions(
        predictor_model_id=args.predictor_model_id,
        querier_model_id=args.querier_model_id,
        n_iterations=args.n_iterations,
        n_ent_samples=args.n_ent_samples,
        class_names=class_names,
        thresholds=thresholds,
        alpha=args.alpha,
        query_bank=query_bank,
        query_bank_answers=query_bank_answers,
        
    )
    for class_name in class_names:
        print(f"LABEL: {class_name}")
        if os.path.exists(f'{save_path}/seed{args.seed}-{class_name}.json'):
            print(f"Already done for {class_name}. Skipping.")
            continue
        results = game.run_one_sample(class_name)
        save_run(f'{save_path}/seed{args.seed}-{class_name}.json', results)

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_iterations', type=int, default=20)
    parser.add_argument('--n_ent_samples', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--threshold_path', type=str)
    parser.add_argument('--predictor_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--querier_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--qry_ans_path', type=str)
    parser.add_argument('--save_dir', type=str, default='./results/thresholds/20q/llama-3.1-8b/')
    return parser.parse_args()

if __name__ == '__main__':
    args = parseargs()
    main(args)
        