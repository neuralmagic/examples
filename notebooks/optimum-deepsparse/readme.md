# Optimum DeepSparse

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuralmagic/notebooks/blob/main/notebooks/optimum-deepsparse/optimum-deepsparse.ipynb)

This notebook illustrates how to run Hugging Face models using DeepSparse Inference Runtime.

[Optimum DeepSparse](https://github.com/neuralmagic/optimum-deepsparse) is a library that enables acceleration of Hugging Face models on CPUs using the [DeepSparse Inference Runtime](https://github.com/neuralmagic/deepsparse). 

```python
from transformers import AutoTokenizer, pipeline
from optimum.deepsparse import DeepSparseModelForTokenClassification

model_id = "hf-internal-testing/tiny-random-RobertaModel"
seq_length = 128
input_shapes = f"[1,{seq_length}]"
model = DeepSparseModelForTokenClassification.from_pretrained(
    model_id,
    export=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
)

outputs = pipe("Norway is beautiful and has great hotels")
print(outputs)
"""
 [{'entity': 'LABEL_0', 'score': 0.5486582, 'index': 1, 'word': 'N', 'start': 0, 'end': 1}, {'entity': 'LABEL_0', 'score': 0.5320723, 'index': 2, 'word': 'or', 'start': 1, 'end': 3}, {'entity': 'LABEL_0', 'score': 0.5137728, 'index': 3, 'word': 'way', 'start': 3, 'end': 6}, {'entity': 'LABEL_1', 'score': 0.5307435, 'index': 4, 'word': 'Ġis', 'start': 7, 'end': 9}, {'entity': 'LABEL_0', 'score': 0.55689734, 'index': 5, 'word': 'Ġbe', 'start': 10, 'end': 12}, {'entity': 'LABEL_0', 'score': 0.5694903, 'index': 6, 'word': 'a', 'start': 12, 'end': 13}, {'entity': 'LABEL_0', 'score': 0.5257641, 'index': 7, 'word': 'ut', 'start': 13, 'end': 15}, {'entity': 'LABEL_1', 'score': 0.5095042, 'index': 8, 'word': 'if', 'start': 15, 'end': 17}, {'entity': 'LABEL_1', 'score': 0.51480544, 'index': 9, 'word': 'ul', 'start': 17, 'end': 19}, {'entity': 'LABEL_1', 'score': 0.54430664, 'index': 10, 'word': 'Ġand', 'start': 20, 'end': 23}, {'entity': 'LABEL_0', 'score': 0.5196867, 'index': 11, 'word': 'Ġhas', 'start': 24, 'end': 27}, {'entity': 'LABEL_1', 'score': 0.51335806, 'index': 12, 'word': 'Ġg', 'start': 28, 'end': 29}, {'entity': 'LABEL_0', 'score': 0.53177285, 'index': 13, 'word': 'reat', 'start': 29, 'end': 33}, {'entity': 'LABEL_0', 'score': 0.52305484, 'index': 14, 'word': 'Ġh', 'start': 34, 'end': 35}, {'entity': 'LABEL_0', 'score': 0.52853996, 'index': 15, 'word': 'ot', 'start': 35, 'end': 37}, {'entity': 'LABEL_1', 'score': 0.5063766, 'index': 16, 'word': 'el', 'start': 37, 'end': 39}, {'entity': 'LABEL_0', 'score': 0.5194421, 'index': 17, 'word': 's', 'start': 39, 'end': 40}]
"""

```
