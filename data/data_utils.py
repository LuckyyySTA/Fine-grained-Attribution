import os
import re
import json

from tqdm import tqdm
import pdb

TASK_INST = {
    "long-form": "Extract the relevant content from the provided documents and then use the extracted content to guide answer generation and cite the sources properly.",
    "short-form": "Extract the relevant content from the provided documents and then use the extracted content to provide a list of accurate answers for the given question. Always cite one and only one document for each answer. Separate answers by commas."
}

def format_sft_data():
    data_path_list = ["./raw_data/long-form.json", "./raw_data/short-form.json"]
    task_instruction_list = [TASK_INST["long-form"], TASK_INST["short-form"]]
    save_path = "./sft.json"

    results = []
    for data_path, task_instruction in zip(data_path_list, task_instruction_list):
        data = json.load(open(data_path))
        for item in data:
            question = item["query"]
            grounding = item["grounding"]
            answer = "[GROUNDING]" + grounding + "[ANSWER]" + item["answer"]
            docs = item["docs"]
            
            doc_list = []
            for doc_id, doc in enumerate(docs):
                doc_text = doc["text"]
                doc_prompt = f"Document [{doc_id+1}]: {doc_text}"
                doc_list.append(doc_prompt)
            
            doc_prompt = "\n".join(doc_list)
            input = f"Question: {question}\n\n{doc_prompt}"

            ret = {"instruction": task_instruction, "input": input, "output": answer}
            results.append(ret)
        
    json.dump(results, open(save_path, 'w'), indent=4)


if __name__ == '__main__':
    format_sft_data()