from vllm import LLM, SamplingParams
from modelscope import snapshot_download

# qwen25_dir = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")
# llm = LLM(qwen25_dir)

llm = LLM("facebook/opt-125m")

prompts = [
    "hello,whats your name"
]
sampling_params = SamplingParams(
    temperature=0,
    max_tokens = 1024
    # top_p=0.95,
)

batch_ans = llm.generate(prompts, sampling_params)

for output in batch_ans:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"输入: {prompt!r}, 大模型给出的输出: {generated_text!r}")
