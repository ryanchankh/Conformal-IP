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
    read_class_names,
    add_to_prompt,
    read_txt,
    read_attr,
    save_json,
    read_json

)
from prompts import (
    QUERY_ANSWERER_SYSTEM_PROMPT,
    PREDICTOR_SYSTEM_PROMPT,
    PREDICTOR_PROMPT,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)

from together import Together


class TwentyQuestions:
    def __init__(
        self,
        predictor_model_id: str,
        n_cal_samples: int = 128,
        class_names: List[str] = None,
        query_bank: List[str] = None,
        query_bank_answers: Dict[str, Dict[str, List[str]]] = None,
        seed: int = 0,
    ):
        self.predictor_model_id = predictor_model_id
        self.n_cal_samples = n_cal_samples
        self.class_names = class_names
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
 
    def get_response(self, model_id, messages):
        response_obj = self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                seed=self.seed
        )
        response = response_obj.choices[0].message.content
        return response

    def sample_length_k_history(self, n_samples, n_queries_to_sample):
        # sample classes
        label_indices = torch.multinomial(torch.ones(self.n_classes), n_samples, replacement=True)

        histories = []
        for label_idx in tqdm(label_indices, desc=f'Sampling history with length {n_queries_to_sample}'):
            # sample k queries
            label = self.class_names[label_idx]
            queries = torch.multinomial(torch.ones(len(self.query_bank)), n_queries_to_sample, replacement=False)
            query_txts = [self.query_bank[q] for q in queries]
            query_answers = [(query_txt, self.get_query_answer(label, query_txt)) \
                for query_txt in query_txts]
            histories.append(query_answers)
        return histories, label_indices
            
    def get_query_answer(self, label, query):
        try:
            possible_answers = self.query_bank_answers[label.item()][query.item()]
        except Exception as e:
            print(f"Error: {e}")
            print(f"Label: {label.item()}, Query: {query.item()}")
            raise e
        answer = np.random.choice(possible_answers, None)
        return answer

    def compute_posteriors(self, histories):
        """Compute posterior from classifier."""
        posteriors = []
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
            outputs = self.predictor_model.generate(**inputs_token, max_new_tokens=1, output_logits=True, return_dict_in_generate=True, pad_token_id=self.predictor_tokenizer.eos_token_id)
            batch_posteriors = self.get_class_probs_from_logits(outputs['logits'][0].cpu())
            posteriors.append(batch_posteriors)
        return torch.vstack(posteriors)

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

    def compute_threshold_for_length_k(self, n_samples, length_k, alphas):
        cal_history, cal_labels = self.sample_length_k_history(n_samples, length_k)        
        cal_posteriors = self.compute_posteriors(cal_history)

        n_cal_samples, _ = cal_posteriors.shape
        assert n_cal_samples != 0, 'cannot have no calibration samples.'
        cal_scores = -cal_posteriors[np.arange(n_cal_samples), np.array(cal_labels)].cpu().numpy()
        
        # 2: get adjusted quantile
        alpha_qhats = {}
        for alpha in alphas:
            q_level = (1 - alpha)
            qhat = np.quantile(cal_scores, q_level, method='higher')
            alpha_qhats[alpha] = qhat.item()
        return alpha_qhats
        
def save_run(path, results):
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)

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
    os.makedirs(args.save_dir, exist_ok=True)

    # class names
    class_names = read_class_names('data/Animals_with_Attributes2/classes20.txt')
    # query_bank = np.array(read_txt('data/Animals_with_Attributes2/query_bank_open/queries.txt'))
    query_bank = np.array(read_txt('data/Animals_with_Attributes2/query_bank_closed/queries.txt'))
    
    query_bank_answers = {cls_name: read_json(os.path.join(args.qry_ans_path, f'{cls_name}.json')) \
        for cls_name in class_names}
    
    # conformal
    print(class_names)
    print(query_bank)

    # generate thresholds
    game = TwentyQuestions(
        predictor_model_id=args.predictor_model_id,
        n_cal_samples=args.n_cal_samples,
        seed=args.seed,
        class_names=class_names,
        query_bank=query_bank,
        query_bank_answers=query_bank_answers,
    )
    alphas =  [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    # alphas = [0.15, 0.2]
    length_alpha_qhats_dict = {}
    for length_k in range(1, args.n_iterations+1):
        alpha_qhats_dict = game.compute_threshold_for_length_k(args.n_cal_samples, length_k, alphas)
        length_alpha_qhats_dict[length_k] = alpha_qhats_dict
    
    # save results
    save_path = os.path.join(args.save_dir, f'seed{args.seed}-n_iterations{args.n_iterations}-n_cal_samples{args.n_cal_samples}.json')
    save_json(save_path, length_alpha_qhats_dict)
    print(save_path)
        
def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_iterations', type=int, default=20)
    parser.add_argument('--n_cal_samples', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='./results/thresholds/20q/llama-3.1-8b/')
    parser.add_argument('--predictor_model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--qry_ans_path', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    args = parseargs()
    main(args)
        