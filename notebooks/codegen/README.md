# Codegen Notebook

## Introduction
This notebook provides a comprehensive overview of the research conducted on the Codegen models, their optimization techniques, evaluation metrics, and performance benchmarks. The focus of this research was to export models to ONNX format, perform quantization and pruning, and evaluate the impact on various metrics, including perplexity. 

## Export to ONNX Format
Two Codegen models were exported to ONNX format: `Salesforce/Codegen-350M-mono` and `Salesforce/Codegen-350M-multi`. The following commands were used for exporting these models:

For `Codegen-350M-mono`:
```bash
git clone https://huggingface.co/Salesforce/codegen-350M-mono
sparseml.transformers.export_onnx \
  --model_path ./codegen-350M-mono \
  --task text-generation \
  --sequence_length 256
```

For `Codegen-350M-multi`:
```bash
git clone https://huggingface.co/Salesforce/codegen-350M-multi
sparseml.transformers.export_onnx --model_path codegen-350M-multi --task text-generation --sequence_len <seq_len>
```


ONNX Export:


