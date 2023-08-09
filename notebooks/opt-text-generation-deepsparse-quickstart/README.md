# Export LLMs directly from HuggingFace for Text Generation with DeepSparse

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuralmagic/notebooks/blob/main/notebooks/opt-text-generation-deepsparse-quickstart/OPT_Text_Generation_DeepSparse_Quickstart.ipynb)

Welcome to our guide on harnessing the power of HuggingFace and DeepSparse! Dive into this notebook to explore the seamless process of converting Large Language Models (LLMs) from HuggingFace to the ONNX format using Optimum. Moreover, we'll supercharge our ONNX model with Key-Value (KV) caching, ensuring faster token generation. Finally, we'll generate captivating text using the DeepSparse engine.

## Key Features:
- **Easy ONNX Exportation:** Convert your favorite OPT models on HuggingFace to ONNX effortlessly.
- **Enhanced Performance:** Integrate KV caching for an accelerated token generation experience.
- **DeepSparse Text Generation:** Harness DeepSparse to create mesmerizing text from your ONNX models.
- **Customizable Settings:** Adapt the notebook to your preferred models, task types, and output directories.

## What's Inside:
1. **Model Configuration & Export:** Set up and export the `facebook/opt-125m` model (or [any OPT model of your choice](https://huggingface.co/models?other=opt&sort=trending&search=facebook%2Fopt)) to ONNX format. Configuration parameters are adjustable based on your desired model and task type.
2. **KV Caching:** Boost your ONNX model's performance with KV caching. Ensure your chosen model supports this feature!
3. **Text Generation with DeepSparse:** Generate text using the DeepSparse engine. Customize input prompts, control output length, and more!

Embark on this journey with us to merge the power of HuggingFace and DeepSparse, making your text generation tasks faster and more efficient. Dive in now! 🚀
