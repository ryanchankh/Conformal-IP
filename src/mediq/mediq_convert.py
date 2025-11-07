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
    extract_and_parse_dict,
    format_predictor_messages,
    format_querier_messages,
    format_query_answerer_messages,

)
from prompts import (
    QUESTION_TO_FACT_SYSTEM_PROMPT,
    QUESTION_TO_FACT_PROMPT,
)

class MediQ:
    def __init__(
        self,
        predictor_model_id: str,
        seed: int = 0,
    ):
        self.predictor_model_id = predictor_model_id
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
            # quantization_config=quantization_config,
            device_map="auto",
        )
        self.predictor_model = AutoModelForCausalLM.from_pretrained(
            self.predictor_model_id,
            trust_remote_code=True,
            torch_dtype='auto',
            attn_implementation="flash_attention_2",
            # quantization_config=quantization_config,
            device_map="auto",
        )
        self.predictor_tokenizer.pad_token = self.predictor_tokenizer.eos_token
        self.predictor_pipeline = pipeline(
            'text-generation',
            model=self.predictor_model,
            tokenizer=self.predictor_tokenizer
        )

    def generate_question(self, fact):
        """Generates a question from a given fact and context using the LLM."""
        messages = [
            {"role": "system", "content": QUESTION_TO_FACT_SYSTEM_PROMPT},
            {"role": "user", "content": QUESTION_TO_FACT_PROMPT.format(fact=fact)}
        ]
        response = self.predictor_pipeline(
            messages,
            max_new_tokens=200
        )
        return response[0]['generated_text'][-1]['content']
    
    def run_one_sample(self, sample_dict):
        facts = eval(sample_dict['facts'])

        facts_lst = []
        questions_lst = []
        for fact in facts:
            question = self.generate_question(fact)
            print(question, fact)
            questions_lst.append(question)
            facts_lst.append(fact)

        results_dict = {
            'facts': facts_lst,
            'questions': questions_lst,
        }
        return results_dict
    
  
def main(args):
    
    # save dir 
    save_dir = f"{args.save_dir}/split{args.split}/"
    os.makedirs(save_dir, exist_ok=True)

    # load datsts            
    mediqa_dataset = MediQDataset('./data/mediQ/', args.specialty, args.split)
    train_dataset, test_dataset, cal_dataset = mediqa_dataset.load()

    # play game
    game = MediQ(predictor_model_id=args.predictor_model_id)
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

    iterator = tqdm(enumerate(train_dataset), total=len(train_dataset), desc='train samples')
    for sample_i, train_sample_dict in iterator:
        save_path = os.path.join(save_dir, f'train_sample{sample_i}.json')
        
        # skip if path exists
        if os.path.exists(save_path):
            print(f"Already done for {sample_i}. Skipping.")
            continue
        
        # run one sample
        results = game.run_one_sample(train_sample_dict)
        results['sample_dict'] = train_sample_dict
        results['params'] = vars(args)
        save_run(save_path, results)

    iterator = tqdm(enumerate(cal_dataset), total=len(cal_dataset), desc='cal samples')
    for sample_i, cal_sample_dict in iterator:
        save_path = os.path.join(save_dir, f'cal_sample{sample_i}.json')
        
        # skip if path exists
        if os.path.exists(save_path):
            print(f"Already done for {sample_i}. Skipping.")
            continue
        
        # run one sample
        results = game.run_one_sample(cal_sample_dict)
        results['sample_dict'] = cal_sample_dict
        results['params'] = vars(args)
        save_run(save_path, results)


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_iterations', type=int, default=20)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--specialty', type=str, required=True)
    parser.add_argument('--predictor_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--save_dir', type=str, default='./results/mediq/convert/llama-3.1-8b/')
    return parser.parse_args()

if __name__ == '__main__':
    args = parseargs()
    main(args)
        