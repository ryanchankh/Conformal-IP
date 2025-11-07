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
        model_id: str,
        huggingface: bool = False
    ):
        self.model_id = model_id
        self.huggingface = huggingface
        self.setup_models()

    def setup_models(self):
        if self.huggingface:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype='auto',
                attn_implementation="flash_attention_2",
                # quantization_config=quantization_config,
                device_map="auto",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype='auto',
                attn_implementation="flash_attention_2",
                # quantization_config=quantization_config,
                device_map="auto",
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pipeline = pipeline(
                'text-generation',
                model=self.model,
                tokenizer=self.tokenizer
            )
        else:
            # APIs
            self.client = Together()
        
    def get_response(self, messages, seed):
        if self.huggingface:
            response = self.pipeline(
                messages,
                max_new_tokens=50,
                temperature=0.5,
                do_sample=True
            )
            response = response[0]['generated_text'][-1]['content']
        else:
            response_obj = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=0.5, 
                    seed=seed
            )
            response = response_obj.choices[0].message.content
        return response

    def get_query_answer(self, label, query, seed):
        messages = format_query_answerer_messages(label, query)
        response = self.get_response(messages, seed)
        return response

def format_query_answerer_messages(label, query):
    messages = [
        {"role": "system", "content": QUERY_ANSWERER_SYSTEM_PROMPT.format(label=label)},
        {"role": "user", "content": query}
    ]
    return messages

def main(args):
    
    # save dir 
    save_path = f'{args.save_dir}/n_answers_per_query{args.n_answers_per_query}/'
    os.makedirs(save_path, exist_ok=True)

    # class names
    class_names = read_class_names('data/Animals_with_Attributes2/classes20.txt')
    query_bank = read_txt('data/Animals_with_Attributes2/query_bank_closed/queries.txt')
    print(class_names)

    # play game
    game = TwentyQuestions(
        model_id=args.query_answerer_model_id,
        huggingface=args.huggingface
    )
    query_answers = {}
    for c, class_name in enumerate(class_names):
        print(f"LABEL: {class_name}")
        if os.path.exists(f'{save_path}/{class_name}.json'):
            print("sample exist.")
            continue

        for query in tqdm(query_bank, total=len(query_bank), desc=f'class: {c} - {class_name}'):
            query_answers[query] = []
            for s in range(args.n_answers_per_query):
                answer = game.get_query_answer(class_name, query, seed=s)
                query_answers[query].append(answer)
                print(f"QUERY: {query}\tANSWER: {answer}", flush=True)
        save_json(f'{save_path}/{class_name}.json', query_answers)

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_answers_per_query', type=int, default=20)
    parser.add_argument('--query_answerer_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--huggingface', default=False, action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results/conformal/entity-deduce-20/llama-3.1-8b/')
    return parser.parse_args()

if __name__ == '__main__':
    args = parseargs()
    main(args)
        