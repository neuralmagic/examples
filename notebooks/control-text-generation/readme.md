# How to Control Text Generation with DeepSparse

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuralmagic/notebooks/blob/main/notebooks/control-text-generation/How-to-Control-Text-Generation-With-DeepSparse.ipynb)

This notebook walks through different strategies for generating text using DeepSparse on CPUs.

## Installation

To install the necessary libraries, run the following command:

```bash
pip install deepsparse-nightly[llm] langchain sentence-transformers chromadb datasets
```

## Generate Text
```python
from deepsparse import TextGeneration

# Define the model path and create a text generation pipeline
MODEL_PATH = "hf:neuralmagic/mpt-7b-chat-pruned50-quant"
text_pipeline = TextGeneration(model_path=MODEL_PATH, sequence_length=2048)

# Configure generation parameters
generation_config = {"top_k": 50, "max_new_tokens": 300}

# Generate text based on a prompt
result = text_pipeline(
    prompt="Translate the following sentence to French `Today is a good day to go out and play football because it is sunny. After that, you can consider visiting the national park for a nature walk while seeing some wild animals.`",
    generation_config=generation_config,
)
print(result.generations[0].text)

"""Il est bon de sortir et jouer au football parce qu’il est jour de soleil. Après cela, il est possible de visiter le parc national pour une balade dans la nature où il est possible de rencontrer certes animaux sauvés"""
```

For more detailed examples and parameter explanations, please refer to the notebook.

## Understanding Generation Parameters
[DeepSparse](https://github.com/neuralmagic/deepsparse/) supports different [text generation parameters](https://github.com/neuralmagic/deepsparse/blob/main/src/deepsparse/transformers/text_generation.md), including:

### Temperature

Temperature is a hyperparameter that controls the randomness of predictions by scaling the logits before applying softmax. When set to 1, the model behaves normally, sampling each word according to its probability. Lower temperatures lead to less randomness and more confident outputs, while higher temperatures encourage diversity and creativity in the text generated.

- **High Temperature (e.g., >1):** The model's outputs become more random and potentially more creative. It's like heating the decision space - more words get a chance to be chosen, even those with lower initial probabilities.
- **Low Temperature (e.g., <1):** The model's outputs become more deterministic. Lower temperatures effectively sharpen the distribution, making the model more conservative and more likely to repeat the most probable sequences of words.

### Top K

The `top_k` sampling parameter restricts the model's choice to the K most likely next words. Setting `top_k` to 50, for example, means that the model only considers the top 50 words sorted by probability to continue the sequence for each step in the generation.

- **Pros:** By constraining the model's choices, `top_k` sampling often leads to more coherent and contextually appropriate text.
- **Cons:** It can exclude potentially fitting choices, especially when the set K is small, limiting creativity and variability in scenarios like storytelling or poetry generation.

### Top P (Nucleus Sampling)

Nucleus sampling, or `top_p` sampling, dynamically determines the number of words to consider by choosing from the smallest set whose cumulative probability exceeds the threshold P. This means it looks at the top probabilities and picks enough words to cover the cumulative probability of P.

For example, if `top_p` is set to 0.8, the model will sum the probabilities from the highest down until it adds up to 0.8, and then sample only from this subset of words.

- **Pros:** This approach allows for dynamic variability, balancing the randomness and determinism based on the actual probability distribution of the next word.
- **Cons:** It may occasionally include very improbable words if they are part of the cumulative set that reaches the desired probability threshold.

### Repetition Penalty

The repetition penalty parameter helps prevent the model from repeating the same words and phrases, enhancing the text's readability and originality. A penalty of 1.0 means no penalty is applied, and as the value increases, the model becomes less likely to repeat recent words.

Applications:
- **Creative Writing:** Increasing the repetition penalty can help produce more diverse and interesting text by discouraging the model from reusing the same language.
- **Informational Text:** A lower or no repetition penalty may be appropriate when the repetition of certain terms is necessary for clarity or emphasis.

By tuning these parameters, you can steer the text generation process to produce outputs that are aligned with your specific goals, whether that be creating novel content or generating precise and informative text.
