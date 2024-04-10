# Optimizing LLMs with One-Shot Pruning and Quantization

This guide delves into optimizing large language models (LLMs) for efficient text generation using neural network compression techniques like sparsification and quantization.
You'll learn how to:

- <b>Sparsify Models:</b> Apply pruning techniques to eliminate redundant parameters from an LLM, reducing its size and computational requirements.
- <b>Quantize Models:</b> Lower the numerical precision of model weights and activations for faster inference with minimal impact on accuracy.
- <b>Evaluate Performance:</b> Measure the impact of sparsification and quantization on model accuracy.

## Prerequisites

- <b>Training Environment:</b> A system that meets the minimum hardware and software requirements as outlined in the [Install Guide](https://docs.neuralmagic.com/get-started/install/#prerequisites).


```python
!pip install "sparseml[transformers]==1.7"
```


## Sparsifying a Llama Model

We'll use a pre-trained, unoptimized [TinyLlama 1.1B chat model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) from the HuggingFace Hub.
The model is referenced by the following stub:
```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

For additional models that work with SparseML, consider the following options:
- Explore pre-sparsified [Generative AI models in the SparseZoo](https://sparsezoo.neuralmagic.com/?modelSet=generative_ai).
- Try out popular LLMs from the [Hugging Face Model Hub](https://huggingface.co/models?pipeline_tag=causal-lm).

### Data Preparation

SparseML requires a dataset to be used for calibration during the sparsification process.
For this example, we'll use the Open Platypus dataset, which is available in the Hugging Face dataset hub and can be loaded as follows:


```python
from datasets import load_dataset

dataset = load_dataset("garage-bAInd/Open-Platypus")
```

### One Shot Compression

Applying pruning and quantization to an LLM without fine-tuning can be done utilizing recipes, the SparseGPT algorithm, and the `compress` command in SparseML.
This combination enables a quick and easy way to sparsify a model, resulting in medium compression levels with minimal accuracy loss, enabling efficient inference.

The code below demonstrates applying one-shot sparsification to the Llama chat model utilizing a recipe.
The recipe specifies using the SparseGPTModifier to apply 50% sparsity and quantization (int8 weights and activations) to the targeted layers within the model.


```python
from sparseml.transformers import (
    SparseAutoModelForCausalLM, SparseAutoTokenizer, compress
)

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

model = SparseAutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = SparseAutoTokenizer.from_pretrained(model_id)

def format_data(data):
    return {
        "text": data["instruction"] + data["output"]
    }

dataset = dataset.map(format_data)

recipe = """
compression_stage:
    run_type: oneshot
    oneshot_modifiers:
        QuantizationModifier:
            ignore: [LlamaRotaryEmbedding, LlamaRMSNorm, SiLUActivation, QuantizableMatMul]
            post_oneshot_calibration: true
            scheme_overrides:
                Linear:
                    weights:
                        num_bits: 8
                        symmetric: true
                        strategy: channel
                Embedding:
                    input_activations: null
                    weights:
                        num_bits: 8
                        symmetric: false
        SparseGPTModifier:
            sparsity: 0.5
            quantize: True
            targets: ['re:model.layers.\d*$']
"""

compress(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    recipe=recipe,
    output_dir="./one-shot-example",
)
```

After running the above code, the model is pruned to 50% sparsity and quantized, resulting in a smaller model ready for efficient inference.

### Inference

To test the model's generation capabilities, we can use the following code to generate text utilizing PyTorch:



```python
from sparseml.transformers import SparseAutoModelForCausalLM, SparseAutoTokenizer
from sparseml.core.utils import session_context_manager

model_path = "./one-shot-example/stage_compression"

with session_context_manager():
  model = SparseAutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0")
tokenizer = SparseAutoTokenizer.from_pretrained(model_path)

chat = [
    {"role": "user", "content": "Tell me about large language models"}
]

inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt").to(model.device)
generated_ids = model.generate(inputs)
outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(outputs)
```

### Evaluating Accuracy

Evaluating the model's accuracy is important to ensure it meets the desired performance requirements.
To do so, we can use the following code to evaluate the model's perplexity on a sample dataset:


```python
from sparseml import evaluate

eval = evaluate(
    "./one-shot-example/stage_compression",
    datasets="openai_humaneval",
    integration="perplexity",
    text_column_name=["prompt", "canonical_solution"]
)
print(eval)
```

The above code, however, does not leverage the sparsity within the model for efficient inference.
To do so, we need to export the model to ONNX to be ready for efficient inference on CPUs with DeepSparse.
SparseML provides a simple export command to do so:


```python
from sparseml import export

export(
    "./one-shot-example/stage_compression",
    task="text-generation",
    sequence_length=1024,
    target_path="./exported"
)
```

The exported model located at `./exported` can now be used for efficient inference with DeepSparse!




```python
!huggingface-cli login
!huggingface-cli upload mgoin/TinyLlama-1.1B-Chat-v1.0-pruned50-quant-ds exported/deployment/
```


## Deploy Sparse LLMs with DeepSparse

[DeepSparse](https://github.com/neuralmagic/deepsparse) is a CPU inference runtime that takes advantage of sparsity to accelerate neural network inference.

LLM inference in DeepSparse is performant with:
* sparse kernels for speedups and memory savings from unstructured sparse weights.
* 8-bit weight and activation quantization support.
* efficient usage of cached attention keys and values for minimal memory movement.

In this section we will explore running the sparse quantized TinyLlama we just made to perform a summarization task.

First, we need to install DeepSparse with LLM dependencies:


```python
!pip install deepsparse[transformers]
```

Next we want to point to our compressed model:


```python
model_path = "exported/deployment/"
```

The task we want to use the LLM for is summarizing some text describing the problem of climate change. Below you can see what the prompt is with the instruction followed by the content to summarize:


```python
text_to_summarize = "Climate change is a global problem that is affecting the planet in numerous ways. Rising temperatures are causing glaciers to melt, sea levels to rise, and weather patterns to become more extreme. These changes are having a significant impact on ecosystems, agriculture, and human health. In order to mitigate the effects of climate change, it is essential to reduce greenhouse gas emissions by transitioning to renewable energy sources, implementing energy-efficient technologies, and encouraging sustainable practices in various sectors such as transportation and agriculture. Additionally, adapting to the inevitable consequences of climate change is crucial, which involves developing resilient infrastructure, improving disaster preparedness, and supporting vulnerable communities. Addressing climate change requires a coordinated global effort from governments, businesses, and individuals to ensure a sustainable future for the planet and its inhabitants."

prompt = f"""
Please summarize the following text, focusing on the key points and main ideas. Keep the summary concise, around 3-5 sentences.

Text:
{text_to_summarize}
"""

print(prompt)
```

Now we will format the prompt to work with the chat template that the model was originally fine-tuned with. You can see in the output from this block what the final input to the model will be before tokenization.


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
chat = [
    {"role": "user", "content": prompt}
]
formatted_prompt = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
print(formatted_prompt)
```

### Pipeline

Now let's plug the model and text into DeepSparse. DeepSparse Pipelines are designed to mirror the Hugging Face Transformers API closely, ensuring a familiar experience if you've worked with Transformers before.
The following code demonstrates how to create a pipeline for text generation using the sparsified LLM you just made:


```python
from deepsparse import TextGeneration

pipeline = TextGeneration(model_path)
result = pipeline(formatted_prompt)

print(result.generations[0].text)
```

The resulting output printed to the console will be the generated text from the model based on the input prompt.

### Server

To make your LLM accessible as a web service, you'll wrap it in a DeepSparse Server.
The Server lets you interact with the model using HTTP requests, making integrating with web applications, microservices, or other systems easy.
DeepSparse Server has an [OpenAI-compatible integration](https://platform.openai.com/docs/api-reference/completions) for request and response formats for seamless integration.

First we need to install the server dependencies with DeepSparse:



```python
!pip install "deepsparse[server]" -qqqqq
```

The following command starts a DeepSparse Server with the sparsified LLM:


```python
!deepsparse.server --integration openai "hf:mgoin/TinyLlama-1.1B-Chat-v1.0-pruned50-quant-ds"
```

With the server running, you can send an HTTP request that conforms to the OpenAI spec to generate text. You can go to http://localhost:5543/docs to learn more about the available endpoints.

Below are examples of using `curl` and `python` to send a request to the server:


```python
import requests
import json

url = "http://localhost:5543/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "hf:mgoin/TinyLlama-1.1B-Chat-v1.0-pruned50-quant-ds",
    "messages": "Large language models are",
    "stream": True
}

response = requests.post(url, headers=headers, data=json.dumps(data))

if response.status_code == 200:
    for chunk in response.iter_content(chunk_size=128):
        print(chunk.decode('utf-8'))  # Decode and print each data chunk
else:
    print("Request failed with status code:", response.status_code)
```


```python
!curl http://localhost:5543/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hf:mgoin/TinyLlama-1.1B-Chat-v1.0-pruned50-quant-ds", "prompt": "Say this is a test", "stream": true}'
```

The resulting output will be the generated text from the model based on the input prompt.



```python

```
