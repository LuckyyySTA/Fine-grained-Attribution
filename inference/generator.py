import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

class LlamaGenerator(object):
    def __init__(self, model_name_or_path):
        self.llm = LLM(model=model_name_or_path, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count())
        self.tokenizer = self.llm.get_tokenizer()
        self.stop_tokens = []

    def batch_data(self, data_list, batch_size):
        n = len(data_list) // batch_size
        batch_data = []
        for i in range(n-1):
            start = i * batch_size
            end = (i+1) * batch_size
            batch_data.append(data_list[start:end])
        last_start = (n-1) * batch_size
        batch_data.append(data_list[last_start:])
        return batch_data

    def generate(self, source, config):
        sampling_params = SamplingParams(
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=config["max_new_tokens"],
            stop=self.stop_tokens,
            skip_special_tokens=config["skip"]
        )
        batch_instances = self.batch_data(source, batch_size=config['batch_size'])
        res_completions = []
        for _, prompt in tqdm(enumerate(batch_instances), total=len(batch_instances), desc="generating"):
            if not isinstance(prompt, list):
                prompt = [prompt]
            completions = self.llm.generate(prompt, sampling_params, use_tqdm=False)

            for output in completions:
                generated_text = [x.text.lstrip('\n').split('\n\n')[0] for x in output.outputs]
                res_completions.append(generated_text)
        return res_completions


MODEL_MAP = {
    "llama": LlamaGenerator
}


def generate_with_llm(model_name_or_path, source, config):
    for token in MODEL_MAP:
        if token in model_name_or_path.lower():
            generator = MODEL_MAP[token](model_name_or_path)
            break
    return generator.generate(source, config)
