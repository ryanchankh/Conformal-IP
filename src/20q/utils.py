import pandas as pd
import numpy as np
import json
import re
import torch


def read_class_names(txt_path):
    class_names = pd.read_csv(txt_path, delimiter='\t', index_col=0, header=None)[2].to_numpy()
    class_names = np.array([name.replace('+', ' ') for name in class_names])
    return class_names

def read_query_bank(predicate_path, predicate_matrix_path, subset_indices):
    attributes = pd.read_csv(predicate_path, delimiter='\t', index_col=0, header=None)[1]
    matrix_binary = pd.read_csv(predicate_matrix_path, delimiter=' ', index_col=None, header=None)
    # matrix_binary = matrix_binary.replace({0: 'No', 1: 'Yes'})
    matrix_binary.columns = np.array(attributes)
    # matrix_binary.index = pd.Index(classes, name='Classes')
    # queries_lst = np.array([f'Is the animal {col}?' for col in matrix_binary.columns])
    queries_lst = np.stack(
        [[f'The animal is not {col}.' for col in matrix_binary.columns],
         [f'The animal is {col}.' for col in matrix_binary.columns]
    ]).T
    return queries_lst, matrix_binary

def read_attr(predicate_matrix_path, subset_indices=None):
    matrix_binary = pd.read_csv(predicate_matrix_path, delimiter=' ', index_col=None, header=None)
    if subset_indices is not None:
        matrix_binary = matrix_binary.iloc[subset_indices]
    return matrix_binary

def read_txt(path):
    """Read a text file line by line and return the lines as a list."""
    lines = []
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

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

def parse_response(response_str):
    """Parse the response from the model."""
    # Split response into lines and remove empty lines
    lines = [line.strip() for line in response_str.split('\n') if line.strip()]
    
    # Extract lines that look like list items (starting with numbers, dashes, or asterisks)
    lst_of_queries = []
    for line in lines:
        # Remove common list markers and leading/trailing whitespace
        cleaned = line.lstrip('0123456789.-*> ').strip()
        if cleaned and cleaned[-1] == '?':  # Only add non-empty lines and questions
            lst_of_queries.append(cleaned)
    return lst_of_queries

def add_qx_to_history(history, query, query_answers):
    history_with_qx = []
    for ans in query_answers:
        ans = ans.replace("(", "").replace(")", "").strip()
        history_with_qx.append(history.copy() + [(query, ans)])
    return history_with_qx

def add_qx_to_histories(histories, query, query_answers):
    histories_wth_qx = []
    for history, qry_ans in zip(histories, query_answers):
        histories_wth_qx += add_qx_to_history(history, query, qry_ans)
    return histories_wth_qx
# def qa_pairs_to_str(qa_pairs_lst):
#     formatted_strs = []
#     for qa_pairs in qa_pairs_lst:
#         formatted_str = ' '.join([f'{qry} {ans}'for qry, ans in qa_pairs])
#         formatted_strs.append(formatted_str)
#     return formatted_strs

def qa_pairs_to_str(qa_pairs_lst):
    formatted_strs = []
    for qa_pairs in qa_pairs_lst:
        formatted_str = ' '.join([f'{ans}'for qry, ans in qa_pairs])
        formatted_strs.append(formatted_str)
    return formatted_strs

def add_to_history(history, role, text):
    history += [{"role": role, "content": text}]
    return history

def add_to_prompt(prompt, role, text):
    prompt += [{"role": role, "content": text}]
    return prompt

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