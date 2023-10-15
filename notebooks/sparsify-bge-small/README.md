# Using Sparsify One-Shot for Sparsifying MiniLM for a Semantic Search Use-Case

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuralmagic/examples/blob/main/notebooks/sparsify-bge-small/Sparsifying_BGE_Small.ipynb
)

In this notebook, we aim to explore the capabilities of the innovative Sparsify one-shot method for quantizing a dense [bge-small](https://huggingface.co/zeroshot/sparse-bge-small-en-v1.5) model, thereby simplifying the DevOps workflow. We will also walk you through the process of abstracting ONNX exportation and optimization by utilizing the [DeepSparse Optimum integration](https://github.com/neuralmagic/optimum-deepsparse). Finally, we will evaluate and compare the accuracy and latency of both the dense and quantized BGE models on only 2 CPU cores in a Colab environment.