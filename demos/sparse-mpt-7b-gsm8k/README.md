# MPT Sparse Finetuned on GSM8k with DeepSparse 
![NM Logo](https://files.slack.com/files-pri/T020WGRLR8A-F05TXD28BBK/neuralmagic-logo.png?pub_secret=54e8db19db)
## Installation 
```bash

pip install requirements.txt
```
## Run App 
```python
gradio app.py

```
![Gradio Demo](gradio.gif)

🚀 **Experience the power of LLM mathematical reasoning** through [our MPT sparse finetuned](https://arxiv.org/abs/2310.06927) on the [GSM8K dataset](https://huggingface.co/datasets/gsm8k). 
GSM8K, short for Grade School Math 8K, is a collection of 8.5K high-quality linguistically diverse grade school math word problems, designed to challenge question-answering systems with multi-step reasoning. 
Observe the model's performance in deciphering complex math questions, such as "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?" and offering detailed step-by-step solutions.

## Accelerated Inference on CPUs 
The MPT model runs purely on CPU courtesy of [sparse software execution by DeepSparse](https://github.com/neuralmagic/deepsparse/tree/main/research/mpt). 
DeepSparse provides accelerated inference by taking advantage of the MPT model's weight sparsity to deliver tokens fast!
![Speedup](https://cdn-uploads.huggingface.co/production/uploads/60466e4b4f40b01b66151416/qMW-Uq8xAawhANTZYB7ZI.png)
