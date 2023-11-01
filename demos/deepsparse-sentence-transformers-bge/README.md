# DeepSparse BGE

This example shows how to run inference with DeepSparse for the BGE models using a new DeepSparse Optimum pathway.

## Optimizing

Neural Magic's DevRel team has created quantized versions of BGE models using our one-shot pathways. We will be following up with some examples of how to quantize additional models.

For now, the quantized models can be found on the Hugging Face Hub:
- [zeroshot/bge-small-en-v1.5-quant](https://huggingface.co/zeroshot/bge-small-en-v1.5-quant)
- [zeroshot/bge-base-en-v1.5-quant](https://huggingface.co/zeroshot/bge-base-en-v1.5-quant)
- [zeroshot/bge-large-en-v1.5-quant](https://huggingface.co/zeroshot/bge-large-en-v1.5-quant)

## Installing

Our optimum integration will be available in our 1.6 release. For the time being install by cloning DeepSparse. 

```bash
git clone https://github.com/neuralmagic/deepsparse.git
pip install -e "./deepsparse[sentence-transformers]"
```

To run the evaluations with MTEB, install MTEB as well.

```bash
pip install mteb
```

> Note: you may need to run `sudo apt install build-essential python3-dev` for MTEB to install properly.

## Running Inference

Checkout [the Jupyter notebook](example.ipynb) for example usage.
