import os
import json
import argparse
from tqdm import tqdm
from generator import generate_with_llm
import pdb

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    )
}

TASK_INST = {
    "long-form": "Extract the relevant content from the provided documents and then use the extracted content to guide answer generation and cite the sources properly.",
    "short-form": "Extract the relevant content from the provided documents and then use the extracted content to provide a list of accurate answers for the given question. Always cite one and only one document for each answer. Separate answers by commas."
}

def deal_with_special_token(data_path):
    data_name = data_path.split(".json")[0]
    dump_path = f"{data_name}_post.json"
    result = json.load(open(data_path))
    
    no_answer_num = 0
    answer_list = []

    for item_idx, item in enumerate(result):
        output_with_grounding = item["output"]
        item["output_with_grounding"] = output_with_grounding
        if "[ANSWER]" not in output_with_grounding:
            no_answer_num += 1
            continue
        else:
            answer_index = output_with_grounding.find("[ANSWER]")
            answer_text = output_with_grounding[answer_index + len("[ANSWER]"):][:-4].strip()
            item["output"] = answer_text
            answer_list.append(item)

    json.dump(answer_list, open(dump_path, 'w'), indent=4)

    print(f"Skip {no_answer_num} instances for evaluation.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="../training/checkpoints/llama-2-7b-stage1")
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--dataset', type=str, default="eli5")
    parser.add_argument('--save_dir', type=str, default="../results")
    parser.add_argument('--dpo', action='store_true')
    parser.add_argument('--skip', action='store_true')

    args = parser.parse_args()

    if args.dpo:
        args.save_dir = os.path.join(args.save_dir, "dpo")

    os.makedirs(args.save_dir, exist_ok=True)

    model_name = args.model.split('/')[-1]
    save_path = os.path.join(args.save_dir, f"{model_name}-{args.dataset}-top-p-{args.top_p}-tmp-{args.temperature}.json")
    config = {
        "n": 1,
        "stop": None,
        "timeout": 15,
        "parallel": True,
        "batch_size": 128,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "skip": args.skip
    }
    print(f"save path: {save_path}")

    if args.dataset == "eli5":
        eval_data_path = "../data/eli5_eval_bm25_top100.json"
        task_instruction = TASK_INST["long-form"]
    elif args.dataset == "asqa":
        eval_data_path = "../data/asqa_eval_gtr_top100.json"
        task_instruction = TASK_INST["long-form"]
    elif args.dataset == "qampari":
        eval_data_path = "../data/qampari_eval_gtr_top100.json"
        task_instruction = TASK_INST["short-form"]

    eval_data = json.load(open(eval_data_path))

    prompt_list = []
    prompt_tempalate = PROMPT_DICT["prompt_input"]

    print(f"task instruction: {task_instruction}")

    for idx, eval_item in enumerate(tqdm(eval_data)):
        question = eval_item["question"]
        eval_item["docs"] = eval_item["docs"][:5]
        docs = eval_item["docs"]
        
        doc_list = []
        for doc_id, doc in enumerate(docs):
            title = doc["title"]
            text = doc["text"]
            doc_text = f"Document [{doc_id+1}](Title: {title}): {text}"
            doc_list.append(doc_text)
        doc_prompt = "\n".join(doc_list)
        
        input_prompt = f"Question: {question}\n\n{doc_prompt}"
        
        meta_data = {"instruction": task_instruction, "input": input_prompt}
        prompt = prompt_tempalate.format_map(meta_data)
        prompt_list.append(prompt)
        eval_item["prompt"] = prompt

    prediction_list = generate_with_llm(args.model, prompt_list, config)

    for d, p in zip(eval_data, prediction_list):
        d["output"] = p[0].strip() if p[0] else ""

    json.dump(eval_data, open(save_path, 'w'), indent=4)

    if not args.skip:
        deal_with_special_token(save_path)
