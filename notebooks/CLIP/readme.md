# CLIP Inference Pipelines

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neuralmagic/examples/blob/main/notebooks/CLIP/CLIP.ipynb)

This notebook demonstrates how to perfom zero-shot image classification and image caption generation on a CPU with [DeepSparse](https://github.com/neuralmagic/deepsparse) and [CLIP](https://github.com/mlfoundations/open_clip/tree/main). 

Follow these steps to run the tasks successfully: 

Step 1: Install [SparseML](https://github.com/neuralmagic/sparseml) with the CLIP option to ensure installation of `open_clip_torch==2.20.0`. 
```bash 
pip install sparseml-nightly[clip]
```

Step 2: Set `torch` to today's nightly version
```python
import os
os.environ["MAX_TORCH"] = "2.2.0.dev20230911+cpu"
```

Step 3: Install DeepSparse with the CLIP option
```bash
pip install deepsparse-nightly[clip]
```

Step 4: Unistall `torch` to install nightly as required by OpenCLIP
```BASH
pip uninstall -y  torch
```

Step 5: Install `torch-nightly`
```bash 
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/
```

After running these stsps succesfully, download sample images for inference:

```bash
wget -O basilica.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg
wget -O buddy.jpeg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/tests/deepsparse/pipelines/sample_images/buddy.jpeg
wget -O thailand.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolact/sample_images/thailand.jpg
```
![basilica](https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg
)
![dog](https://raw.githubusercontent.com/neuralmagic/deepsparse/main/tests/deepsparse/pipelines/sample_images/buddy.jpeg
)

![dog](https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolact/sample_images/thailand.jpg)

Download the scripts for exporting the CLIP models as ONNX files:
```bash
wget https://raw.githubusercontent.com/neuralmagic/sparseml/main/integrations/clip/clip_models.py
wget https://raw.githubusercontent.com/neuralmagic/sparseml/main/integrations/clip/clip_onnx_export.py
```

Export a zero-shot image classification model:
```bash
python clip_onnx_export.py --model convnext_base_w_320 \
            --pretrained laion_aesthetic_s13b_b82k --export-path convnext_onnx
```
Perform zero-shot image classification:

```python
import numpy as np

from deepsparse import BasePipeline
from deepsparse.clip import (
    CLIPTextInput,
    CLIPVisualInput,
    CLIPZeroShotInput
)

possible_classes = ["ice cream", "an elephant", "a dog", "a building", "a church"]
images = ["basilica.jpg", "buddy.jpeg", "thailand.jpg"]

model_path_text = "convnext_onnx/clip_text.onnx"
model_path_visual = "convnext_onnx/clip_visual.onnx"

kwargs = {
    "visual_model_path": model_path_visual,
    "text_model_path": model_path_text,
}
pipeline = BasePipeline.create(task="clip_zeroshot", **kwargs)

pipeline_input = CLIPZeroShotInput(
    image=CLIPVisualInput(images=images),
    text=CLIPTextInput(text=possible_classes),
)

output = pipeline(pipeline_input).text_scores
for i in range(len(output)):
    prediction = possible_classes[np.argmax(output[i])]
    print(f"Image {images[i]} is a picture of {prediction}")

"""
DeepSparse, Copyright 2021-present / Neuralmagic, Inc. version: 1.6.0.20230906 COMMUNITY | (f5e597bf) (release) (optimized) (system=avx2, binary=avx2)
Image basilica.jpg is a picture of a church
Image buddy.jpeg is a picture of a dog
Image thailand.jpg is a picture of an elephant
"""
```

Export the image captioning CLIP models:
```bash
python clip_onnx_export.py --model coca_ViT-B-32 \
            --pretrained mscoco_finetuned_laion2b_s13b_b90k --export-path caption_models
```

Perform image captioning:
```python
from deepsparse import BasePipeline
from deepsparse.clip import CLIPCaptionInput, CLIPVisualInput

root = "caption_models"
model_path_visual = f"{root}/clip_visual.onnx"
model_path_text = f"{root}/clip_text.onnx"
model_path_decoder = f"{root}/clip_text_decoder.onnx"
engine_args = {"num_cores": 8}

kwargs = {
    "visual_model_path": model_path_visual,
    "text_model_path": model_path_text,
    "decoder_model_path": model_path_decoder,
    "pipeline_engine_args": engine_args
}
pipeline = BasePipeline.create(task="clip_caption", **kwargs)

pipeline_input = CLIPCaptionInput(image=CLIPVisualInput(images="thailand.jpg"))
output = pipeline(pipeline_input).caption
print(output[0])

"""
an adult elephant and a baby elephant 
"""
```
## Where to go From Here

Join us on [Slack](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ) for any questions or create an issue on [GitHub](https://github.com/neuralmagic).