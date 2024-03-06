## Performantly Quantize LLMs to 4-bits with Marlin and nm-vllm
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuralmagic/notebooks/blob/main/notebooks/marlin-nm-vllm/Performantly_Quantize_LLMs_to_4_bits_with_Marlin_and_nm_vllm.ipynb)


This notebook walks through how to compress a pretrained LLM and deploy it with `nm-vllm`.

## Installation

To install the necessary libraries, run the following command:

```bash
pip install auto-gptq==0.7.1 torch==2.2.1 
```
## Perform Quantization 
Follow the steps in the notebook to perform quantization. The process involves quantizing the model using GPTQ and then converting it to Marlin format. 

## Deploy With nm-vllm
You can save the resulting model to Hugging Face and use it for inference using `nm-vllm`. 
```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

model_id = "neuralmagic/TinyLlama-1.1B-Chat-v1.0-marlin"
model = LLM(model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
messages = [
    {"role": "user", "content": "How to make banana bread?"},
]
formatted_prompt =  tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
sampling_params = SamplingParams(max_tokens=200)
outputs = model.generate(formatted_prompt, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
"""
Sure! Here's a simple recipe for banana bread:

Ingredients:
- 3-4 ripe bananas,mashed
- 1 large egg
- 2 Tbsp. Flour
- 2 tsp. Baking powder
- 1 tsp. Baking soda
- 1/2 tsp. Ground cinnamon
- 1/4 tsp. Salt
- 1/2 cup butter, melted
- 3 Cups All-purpose flour
- 1/2 tsp. Ground cinnamon

Instructions:

1. Preheat your oven to 350 F (175 C).
"""
```
For more details on how to deploy, go to the [nm-vllm Github repo](https://github.com/neuralmagic/nm-vllm).

For further support, and discussions on these models join [Neural Magic's Slack Community](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ)