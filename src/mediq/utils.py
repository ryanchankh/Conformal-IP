import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
 
import pandas as pd
import numpy as np
import json    
import re

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from prompts import (
    PREDICTOR_SYSTEM_PROMPT,
    PREDICTOR_PROMPT,
    QUERIER_SYSTEM_PROMPT,
    QUERIER_PROMPT,
    QUERY_ANSWERER_SYSTEM_PROMPT,
    QUERY_ANSWERER_PROMPT
    
)
def save_json(path, data):
    import json
    with open(path, 'w') as f:
        json.dump(data, f)

def read_json(path):
    import json
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def save_run(path, results):
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)

def extract_and_parse_dict(text):
    text = text.replace("\n", " ")
    
    # Search for a substring enclosed in curly braces
    match = re.search(r'\{.*?\}', text)
    
    if match:
        extracted = match.group(0)
        try:
            parsed_dict = json.loads(extracted)
            return parsed_dict
        except json.JSONDecodeError:
            # Return empty string if JSON is invalid
            return ""
    else:
        # Return empty string if no match found
        return ""

def compute_entropy_formula(test_posteriors):
    probs = test_posteriors
    assert torch.all(probs >= 0.)
    probs = torch.clamp(probs, min=1e-5)
    return -(probs * torch.log2(probs)).sum(-1).mean()

def compute_predict_sets(test_posteriors, qhat):
    """Compute prediction set."""
    prediction_sets = torch.where(test_posteriors >= (-qhat), 1., 0)
    return prediction_sets.float()

def compute_upper_bound_formula(posteriors, labels, alpha, qhat, label_sizes):
    n_ent_samples = posteriors.shape[0]
    psets = compute_predict_sets(posteriors, qhat)
    include_inds = torch.where(psets[torch.arange(n_ent_samples), labels]==1, True, False)
    include_psets_sizes = psets[include_inds].sum(1)
    exclude_psets_sizes = psets[~include_inds].sum(1)

    n = include_psets_sizes.shape[0] + exclude_psets_sizes.shape[0]
    h_b = -alpha * np.log2(alpha) - (1 - alpha) * np.log2(1 - alpha)
    # alpha_n = alpha - (1 / (n + 1))
    alpha_n = alpha
    if len(include_psets_sizes) == 0:
        include_exp = torch.tensor([0.]).cuda()
    else:
        include_exp = (1 - alpha_n) * torch.log2(include_psets_sizes).mean()
    if len(exclude_psets_sizes) == 0:
        exclude_exp = torch.tensor([0.]).cuda()
    else:
        exclude_exp = alpha * torch.log2(label_sizes - exclude_psets_sizes).mean()
    final_bound = h_b + include_exp + exclude_exp
    return final_bound


def format_predictor_messages(history, question, options, init_info):
    # input prompt
    if len(history) == 0:
        input_prompt = "You have not gathered any information yet."
    else:
        input_prompt = "Here is the information you have gathered.\n"
        for k, (q, qx) in enumerate(history):
            # history_prompt += f"Doctor's Question: {q}\nPatient's Answer: {qx}\n"
            input_prompt += f"Patient's Fact: {qx}\n"
            
    messages = [
        {"role": "system", "content": PREDICTOR_SYSTEM_PROMPT},
        {"role": "user", "content": PREDICTOR_PROMPT.format(
            init_info=init_info, history_lst=input_prompt, question=question, option_a=options['A'],
            option_b=options['B'], option_c=options['C'], option_d=options['D'],
        )},
        {"role": "assistant", "content": """{\"answer\": \""""}
    ]
    return messages

def format_querier_messages(history, init_info, n_queries_per_step):
    if len(history) == 0:
        history_prompt = f"You have not gathered any information yet."
    else:
        history_prompt = "Here is the information you have gathered.\n"
        for k, (q, qx) in enumerate(history):
            # history_prompt += f"Doctor's Question: {q}\nPatient's Answer: {qx}\n"
            history_prompt += f"Patient's Fact: {qx}\n"
    messages = \
        [{
            "role": "system",
            "content": QUERIER_SYSTEM_PROMPT.format(n_queries_per_step=n_queries_per_step)
         },
         {
             "role": "user",
             "content": QUERIER_PROMPT.format(history_lst=history_prompt, init_info=init_info, n_queries_per_step=n_queries_per_step)
        }]
    return messages

def format_query_answerer_messages(facts, query):
    fact_lst = "\n".join(facts)
    messages = [
            {"role": "system", "content": QUERY_ANSWERER_SYSTEM_PROMPT},
            {"role": "user", "content": QUERY_ANSWERER_PROMPT.format(fact_lst=fact_lst, query=query)}
        ]   
    return messages