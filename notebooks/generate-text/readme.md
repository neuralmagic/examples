# How to Generate Text on CPUs With DeepSparse

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuralmagic/notebooks/blob/main/notebooks/generate-text/generate.ipynb)

This notebook walks through different strategies for generating text using DeepSparse on CPUs.

## Installation
```bash
pip install deepsparse-nightly[llm] langchain sentence-transformers chromadb datasets
```
## Generate Text
```python
from deepsparse import TextGeneration

MODEL_PATH = "hf:neuralmagic/mpt-7b-chat-pruned50-quant"

text_pipeline = TextGeneration(model_path=MODEL_PATH, sequence_length=2048)

generation_config = {"top_k": 50, "max_new_tokens": 300}

result = text_pipeline(
    prompt="Translate the following sentence to French `Today is a good day to go out and play football because it is sunny. After that, you can consider visiting the national park for a nature walk while seeing some wild animals.`",
    generation_config=generation_config,
)
print(result.generations[0].text)

"""Il est bon de sortir et jouer au football parce qu’il est jour de soleil. Après cela, il est possible de visiter le parc national pour une balade dans la nature où il est possible de rencontrer certes animaux sauvés"""

```
## Text Generation Parameters
[DeepSparse](https://github.com/neuralmagic/deepsparse/) allows you to set different text generation parameters. 

### Temperature

The temperature parameter is used to modify the logits. The logits are passed to the softmax function to turn them into probabilities, making it easier to pick the next word. 

When the temperature is 1, the operation is the same as that of a normal softmax operation. 

When the temperature is high, the model becomes more random hence associated with being more creative. 

When the temperature is low the model becomes more conservative hence more confident with its responses. 

### Top K
In `top_k` sampling, the user sets the top number of words that the model will sample from. For example, if K is 50, the model will sample from the top 50 words. 

The problem with this is that you have to manually choose the K, meaning some words are left out. This may not be ideal for some use cases such as creative writing.

### Top P
In `top_p` sampling, the value of K is set dynamically by setting a desired probability.

The model will choose the least number of words that exceed the chosen probability, making the number of words dynamic.

For instance, if you pick p as 0.8. The probability of words picked can be 0.5+0.2+0.1 or 0.3+0.3+0.2.

### Repetition Penalty
Repetition penalty is an important parameter that ensures the model doesn't repeat certain words or phrases.

Setting it to 1, means that there is no penalty. For example, in creative writing, you can penalize the model for repeating phrases that recently appeared in the text. 
