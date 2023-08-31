# Using Sparsify One-Shot for Sparsifying MiniLM for a Semantic Search Use-Case

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuralmagic/examples/blob/main/notebooks/sparsify-sentence-embeddings/Sparsify_One-Shot.ipynb
)

In this notebook, we aim to explore the capabilities of the innovative Sparsify one-shot method for quantizing a dense [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model, thereby simplifying the DevOps workflow. We will also walk you through the process of abstracting ONNX exportation and optimization by utilizing the [DeepSparse Optimum integration](https://github.com/neuralmagic/optimum-deepsparse). Finally, we will evaluate and compare the accuracy and latency of both the dense and quantized MiniLM models. To demonstrate their effectiveness, we'll employ the Weaviate vector database to efficiently index and search embeddings, underlining the preservation of MiniLM's semantic search functionalities despite the use of INT8 quantization and one-shot weight pruning.
