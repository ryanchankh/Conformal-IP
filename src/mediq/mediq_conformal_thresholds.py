import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
from typing import List, Dict
import json
import os
from itertools import product
from tqdm import tqdm

import numpy as np
import torch

from utils import (
    save_json,
    read_json,
    save_run

)
from prompts import (
    QUERY_ANSWERER_SYSTEM_PROMPT,
    QUERY_ANSWERER_PROMPT,
    PREDICTOR_SYSTEM_PROMPT,
    PREDICTOR_PROMPT,
)
from mediq_dataset import MediQDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    set_seed
)
from datasets import Dataset



class MediQCalibration:
    def __init__(
        self,
        predictor_model_id: str,
        query_answerer_model_id: str,
        n_iterations: int = 20,
        n_cal_samples: int = 128,
        cal_dataset: Dataset = None,
        specialty: str = None,
        seed: int = 0,
    ):
        self.predictor_model_id = predictor_model_id
        self.query_answerer_model_id = query_answerer_model_id
        self.n_iterations = n_iterations
        self.n_cal_samples = n_cal_samples
        self.specialty = specialty
        self.cal_dataset = cal_dataset
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

    def get_query_answer(self, facts, query):
        messages = format_query_answerer_messages(facts, query)
        response = self.get_response(messages)
        return response

    def get_query_answers(self, facts_lst, queries):
        assert isinstance(facts_lst, List) or isinstance(facts_lst, np.ndarray), 'samples need to be of type List'
        assert isinstance(queries, List) or isinstance(queries, np.ndarray), 'queries need ot be of type List'
  
        query_answers = []
        for fact_i, facts in tqdm(enumerate(facts_lst), total=len(facts_lst), desc='get_query_answers'):
            query_answers_for_label = []
            for query in queries:
                qry_ans = self.get_query_answer(facts, query)
                query_answers_for_label.append(qry_ans)
            query_answers.append(query_answers_for_label)
        return np.array(query_answers)

    def sample_cal_data(self, n_cal_samples):
        sample_indices = np.random.choice(len(self.cal_dataset), n_cal_samples, replace=True)
        questions, options, facts, init_infos, labels, queries = [], [], [], [], [], []
        for sample_idx in sample_indices:
            sample_dict = self.cal_dataset[sample_idx.item()]
            questions.append(sample_dict['question'])
            options.append(eval(sample_dict['options']))
            facts.append(eval(sample_dict['facts']))
            init_infos.append(eval(sample_dict['context'])[0])
            labels.append({"A": 0, "B": 1, "C": 2, "D": 3}[sample_dict['answer_idx']])
            queries.append(sample_dict['queries'])
        return questions, options, facts, init_infos, labels, queries

    def sample_length_k_history(self, n_cal_samples, n_queries_to_sample):
        """Sample a history of length k."""
        
        # sample calibration data
        sampled_questions, sampled_options, sampled_facts, sampled_infos, sampled_labels, sampled_queries \
            = self.sample_cal_data(n_cal_samples)
        
        histories = []
        for facts, queries in zip(sampled_facts, sampled_queries):
            
            # sample k queries
            sampled_queries_indices = np.random.choice(len(queries), n_queries_to_sample, replace=True)
            sampled_queries = [queries[i] for i in sampled_queries_indices]
            
            
            # get query_answers
            query_answers = [facts[q] for q in sampled_queries_indices]
            # query_answers = self.get_query_answers([facts], sampled_queries)
            history = [(qry, qry_ans) for qry, qry_ans in zip(queries, query_answers)]
            print(history)
            # add to history
            histories.append(history)
            
        return histories, sampled_questions, sampled_options, sampled_facts, sampled_infos, sampled_labels

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
            inputs_str = self.predictor_tokenizer.apply_chat_template([messages], add_generation_prompt=False, tokenize=False, continue_chat_template=True)
            if self.predictor_model_id == 'meta-llama/Meta-Llama-3.1-8B-Instruct':
                inputs_str = [inp_str[:-10] for inp_str in inputs_str] # get around bug where continue_chat_template adds extra tokens
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
            class_indices = torch.tensor([32, 33, 34, 35], dtype=torch.long)
        else:
            raise ValueError(f"Model {self.predictor_model_id} not supported.")
        return torch.softmax(logits[:, class_indices], dim=-1)

    def compute_threshold_for_length_k(self, n_samples, length_k, alphas):
        cal_histories, sampled_questions, sampled_options, sampled_facts, sampled_infos, sampled_labels \
            = self.sample_length_k_history(n_samples, length_k)        
        cal_posteriors, _ = self.compute_posteriors(cal_histories, sampled_questions, sampled_options, sampled_infos)

        n_cal_samples, _ = cal_posteriors.shape
        # 1: get conformal scores. n = calib_Y.shape[0]
        assert n_cal_samples != 0, 'cannot have no calibration samples.'
        cal_scores = -cal_posteriors[np.arange(n_cal_samples), np.array(sampled_labels)].cpu().numpy()
        
        # 2: get adjusted quantile
        alpha_qhats = {}
        for alpha in alphas:
            # q_level = np.ceil((n_cal_samples + 1) * (1 - alpha)) / n_cal_samples
            q_level = (1 - alpha)
            qhat = np.quantile(cal_scores, q_level, method='higher')
            alpha_qhats[alpha] = qhat.item()
        return alpha_qhats

def format_predictor_messages(history, question, options, init_info):
    # input prompt
    if len(history) == 0:
        input_prompt = "You have not gathered any information yet."
    else:
        input_prompt = "Here is the information you have gathered.\n"
        for k, (_, qx) in enumerate(history):
            input_prompt += f"{k+1}. {qx}\n"
            
    messages = [
        {"role": "system", "content": PREDICTOR_SYSTEM_PROMPT},
        {"role": "user", "content": PREDICTOR_PROMPT.format(
            init_info=init_info, history_lst=input_prompt, question=question, option_a=options['A'],
            option_b=options['B'], option_c=options['C'], option_d=options['D'],
        )},
        {"role": "assistant", "content": """{\"answer\": \""""}
    ]
    return messages

def format_query_answerer_messages(facts, query):
    fact_lst = "\n".join(facts)
    messages = [
            {"role": "system", "content": QUERY_ANSWERER_SYSTEM_PROMPT},
            {"role": "user", "content": QUERY_ANSWERER_PROMPT.format(fact_lst=fact_lst, query=query)}
        ]   
    return messages

def main(args):
    
    # save dir 
    os.makedirs(args.save_dir, exist_ok=True)

    # load datsts            
    mediqa_dataset = MediQDataset('./data/mediQ/', args.specialty, args.split_idx)
    _, _, cal_dataset = mediqa_dataset.load()
    
    # play game
    game = MediQCalibration(
        predictor_model_id=args.predictor_model_id,
        query_answerer_model_id=args.query_answerer_model_id,
        n_iterations=args.n_iterations,
        n_cal_samples=args.n_cal_samples,
        cal_dataset=cal_dataset,
        specialty=args.specialty,
        seed=args.seed
    )

    # generate thresholds
    length_alpha_qhats_dict = {}
    for length_k in range(1, args.n_iterations+1):
        alpha_qhats_dict = game.compute_threshold_for_length_k(args.n_cal_samples, length_k, args.alphas)
        length_alpha_qhats_dict[length_k] = alpha_qhats_dict
    
        # save results
        save_path = os.path.join(args.save_dir, f'seed{args.seed}-n_iterations{args.n_iterations}-n_cal_samples{args.n_cal_samples}.json')
        save_json(save_path, length_alpha_qhats_dict)
        print(save_path)
        
def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictor_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--query_answerer_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--n_iterations', type=int, default=20)
    parser.add_argument('--n_cal_samples', type=int, default=64)
    parser.add_argument('--alphas', type=float, nargs='*', help="coverage")
    parser.add_argument('--specialty', type=str, required=True)
    parser.add_argument('--split_idx', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./thresholds/mediq/')
    return parser.parse_args()

if __name__ == '__main__':
    args = parseargs()
    main(args)
        