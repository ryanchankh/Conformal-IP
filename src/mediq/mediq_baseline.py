import os
from tqdm import tqdm

from mediq_dataset import MediQDataset
from utils import save_json, read_json
from prompts import (
    BASELINE_SYSTEM_PROMPT,
    BASELINE_USER_PROMPT,
    PREDICTOR_SYSTEM_PROMPT,
    PREDICTOR_PROMPT
)

from together import Together




def format_predictor_messages(history, question, options, init_info):
    # input prompt
    if len(history) == 0:
        input_prompt = "You have not gathered any information yet."
    else:
        input_prompt = "Here is the information you have gathered.\n"
        for k, qx in enumerate(history):
            input_prompt += f"Patient's Fact: {qx}\n"

    messages = [
        {"role": "system", "content": PREDICTOR_SYSTEM_PROMPT},
        {"role": "user", "content": PREDICTOR_PROMPT.format(
            init_info=init_info, history_lst=input_prompt, question=question, option_a=options['A'],
            option_b=options['B'], option_c=options['C'], option_d=options['D'],
        )},
    ]
    return messages

def get_response(client, messages):
    import pdb; pdb.set_trace()
    response_obj = client.chat.completions.create(
            model=args.predictor_model_id,
            messages=messages,
    )
    response = response_obj.choices[0].message.content
    return response

def main(args):
    
    # save dir
    save_dir = f'{args.save_dir}/split{args.split}'
    os.makedirs(save_dir, exist_ok=True)

    # Load MedQA dataset
    mediqa_dataset = MediQDataset('./data/mediQ/', args.specialty, args.split)
    _, test_dataset, _ = mediqa_dataset.load()

    # Load model and tokenizer
    client = Together()
    
    results = []
    for sample_i, sample_dict in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        context = eval(sample_dict['facts'])
        question = sample_dict['question']
        options = eval(sample_dict['options'])
        specialty = sample_dict['specialty']

        save_path = f'{save_dir}/test_sample{sample_i}.json'
        if os.path.exists(save_path):
            continue

        # Format prompt
        messages = format_predictor_messages(context, question, options, specialty)
        response = get_response(client, messages)

        print("MESSAGES:", messages)
        print("RESPONSE:", response)

        results = {
            'response': response,
            'sample_dict': sample_dict,
            'params': vars(args)
        }
        save_json(save_path, results)
        
def parseargs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictor_model_id', type=str, required=True) 
    parser.add_argument('--specialty', type=str, required=True) 
    parser.add_argument('--split', type=int, required=True) 
    parser.add_argument('--save_dir', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parseargs()
    main(args)