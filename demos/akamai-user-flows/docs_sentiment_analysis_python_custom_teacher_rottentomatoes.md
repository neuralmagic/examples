# **Sentiment Analysis: Sparse Transfer Learning with the Python API**

In this example, you will fine-tune a 90% pruned BERT model onto the Rotten Tomatoes dataset with a custom distillation teacher model using SparseML's Hugging Face Integration.

### **Sparse Transfer Learning Overview**

Sparse Transfer Learning is very similiar to the typical transfer learning process used to train NLP models, where we fine-tune a pretrained checkpoint onto a smaller downstream dataset. With Sparse Transfer Learning, however, we simply start the training process from a pre-sparsified checkpoint and maintain sparsity while the fine-tuning occurs.

At the end, you will have a sparse model trained on your dataset, ready to be deployed with DeepSparse for GPU-class performance on CPUs!

### **Pre-Sparsified BERT**
SparseZoo, Neural Magic's open source repository of pre-sparsified models, contains a 90% pruned version of BERT, which has been sparsified on the upstream Wikipedia and BookCorpus datasets with the
masked language modeling objective. [Check out the model card](https://sparsezoo.neuralmagic.com/models/obert-base-wikipedia_bookcorpus-pruned90). We will use this model as the starting point for the transfer learning process.

First, we must install SparseML with Transformers:


```python
!pip install "sparseml[transformers]==1.7"
```

If you are running on Google Colab, restart the runtime after this step.


```python
import sparseml
from sparsezoo import Model
from sparseml.transformers.utils import SparseAutoModel
from sparseml.transformers.sparsification import Trainer, TrainingArguments
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    default_data_collator
)
from datasets import load_dataset, load_metric
```

## **Step 1: Load a Dataset**

SparseML is integrated with Hugging Face, so we can use the `datasets` class to load datasets from the Hugging Face hub or from local files.

[Rotten Tomatoes Dataset Card](https://huggingface.co/datasets/rotten_tomatoes)


```python
# load dataset natively
dataset = load_dataset("rotten_tomatoes")

# alternatively, save to save to csv and reload as example
dataset["train"].to_csv("rotten_tomatoes-train.csv")
dataset["validation"].to_csv("rotten_tomatoes-validation.csv")
data_files = {
  "train": "rotten_tomatoes-train.csv",
  "validation": "rotten_tomatoes-validation.csv"
}
dataset_from_json = load_dataset("csv", data_files=data_files)
```


```python
print(dataset_from_json)
```


```python
!head rotten_tomatoes-train.csv --lines=5
```


```python
# configs for below
INPUT_COL_1 = "text"
INPUT_COL_2 = None
LABEL_COL = "label"
NUM_LABELS = len(dataset_from_json["train"].unique(LABEL_COL))
print(NUM_LABELS)
```

## **Step 2: Setup Evaluation Metric**

Sentiment analysis is a single sequence binary classification problem. We will use the `accuracy` function as the evaluation metric.

Since SparseML is integrated with Hugging Face, we can use the native Hugging Face `compute_metrics` for evaluation (which will be passed to the `Trainer` class below).


```python
metric = load_metric("accuracy")

def compute_metrics(p: EvalPrediction):
  preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
  preds = np.argmax(preds, axis=1)
  result = metric.compute(predictions=preds, references=p.label_ids)
  if len(result) > 1:
      result["combined_score"] = np.mean(list(result.values())).item()
  return result
```

## **Step 3: Download Files for Sparse Transfer Learning**

First, we need to select a sparse checkpoint to begin the training process. In this case, we will fine-tune a 90% pruned version of BERT onto the Rotten Tomatoes dataset. This model is available in SparseZoo, identified by the following stub:
```
zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none
```

Next, we need to create a sparsification recipe for usage in the training process. Recipes are YAML files that encode the sparsity related algorithms and parameters to be applied by SparseML. For Sparse Transfer Learning, we need to use a recipe that instructs SparseML to maintain sparsity during the training process and to apply quantization over the final few epochs.  In SparseZoo, there is a transfer recipe which was used to fine-tune BERT onto the SST2 task. Since Rotten Tomatoes is a similiar problem to SST2, we will use the SST2 recipe, which is identified by the following stub:
```
zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none
```

Finally, SparseML has the optional ability to apply model distillation from a teacher model during the transfer learning process to boost accuracy. Since SparseML is integrated with Hugging Face, we can use a model from the Hugging Face hub. We will use BERT-base trained by textattack on rotten tomatoes as the teacher ([Model Card](https://huggingface.co/textattack/bert-base-uncased-rotten-tomatoes)). It is identified by the following:

```
textattack/bert-base-uncased-rotten-tomatoes
```

Use the `sparsezoo` python client to download the models and recipe using their SparseZoo stubs.


```python
# downloads 90% pruned upstream BERT trained on MLM objective
model_stub = "zoo:nlp/masked_language_modeling/obert-base/pytorch/huggingface/wikipedia_bookcorpus/pruned90-none"
model_path = Model(model_stub, download_path="./model").training.path

# downloads transfer recipe for MNLI (pruned90_quant)
transfer_stub = "zoo:nlp/sentiment_analysis/obert-base/pytorch/huggingface/sst2/pruned90_quant-none"
recipe_path = Model(transfer_stub, download_path="./transfer_recipe").recipes.default.path
```


```python
# https://huggingface.co/textattack/bert-base-uncased-rotten-tomatoes
# this is a model from the huggingface hub, trained on rotten tomatoes
teacher_path = "textattack/bert-base-uncased-rotten-tomatoes"
```

We can see that the upstream model (trained on Wikipedia BookCorpus) and  configuration files have been downloaded to the local directory.


```python
%ls ./model/training
```

#### Inspecting the Recipe

Here is the transfer learning recipe:

```yaml
version: 1.1.0

# General Variables
num_epochs: &num_epochs 13
init_lr: 1.5e-4
final_lr: 0

qat_start_epoch: &qat_start_epoch 8.0
observer_epoch: &observer_epoch 12.0
quantize_embeddings: &quantize_embeddings 1

distill_hardness: &distill_hardness 1.0
distill_temperature: &distill_temperature 2.0

weight_decay: 0.01

# Modifiers:

training_modifiers:
  - !EpochRangeModifier
      end_epoch: eval(num_epochs)
      start_epoch: 0.0
  - !LearningRateFunctionModifier
      start_epoch: 0
      end_epoch: eval(num_epochs)
      lr_func: linear
      init_lr: eval(init_lr)
      final_lr: eval(final_lr)

quantization_modifiers:
  - !QuantizationModifier
      start_epoch: eval(qat_start_epoch)
      disable_quantization_observer_epoch: eval(observer_epoch)
      freeze_bn_stats_epoch: eval(observer_epoch)
      quantize_embeddings: eval(quantize_embeddings)
      quantize_linear_activations: 0
      exclude_module_types: ['LayerNorm', 'Tanh']
      submodules:
        - bert.embeddings
        - bert.encoder
        - bert.pooler
        - classifier


distillation_modifiers:
  - !DistillationModifier
     hardness: eval(distill_hardness)
     temperature: eval(distill_temperature)
     distill_output_keys: [logits]

constant_modifiers:
  - !ConstantPruningModifier
      start_epoch: 0.0
      params: __ALL_PRUNABLE__

regularization_modifiers:
  - !SetWeightDecayModifier
      start_epoch: 0.0
      weight_decay: eval(weight_decay)
```


The `Modifiers` in the transfer learning recipe are the important items that encode how SparseML should modify the training process for Sparse Transfer Learning:
- `ConstantPruningModifier` tells SparseML to pin weights at 0 over all epochs, maintaining the sparsity structure of the network
- `QuantizationModifier` tells SparseML to quanitze the weights with quantization aware training over the last 5 epochs
- `DistillationModifier` tells SparseML how to apply distillation during the trainign process, targeting the logits

Below, SparseML's `Trainer` will parses the modifiers and updates the training process to implement the algorithms specified here.

## **Step 4: Setup Hugging Face Model Objects**

Next, we will set up the Hugging Face `tokenizer, config, and model`.

These are all native Hugging Face objects, so check out the Hugging Face docs for more details on `AutoModel`, `AutoConfig`, and `AutoTokenizer` as needed.

We instantiate these classes by passing the local path to the directory containing the `pytorch_model.bin`, `tokenizer.json`, and `config.json` files from the SparseZoo download.


```python
# we can use a shared tokenizer since both are BERT
# see examples for using a separate tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# setup model configs
model_config = AutoConfig.from_pretrained(model_path, num_labels=NUM_LABELS)
teacher_config = AutoConfig.from_pretrained(teacher_path, num_labels=NUM_LABELS)

# initialize model using familiar HF AutoModel
model_kwargs = {"config": model_config}
model_kwargs["state_dict"], s_delayed = SparseAutoModel._loadable_state_dict(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path,**model_kwargs,)

# initialize teacher using familiar HF AutoModel
teacher_kwargs = {"config": teacher_config}
teacher_kwargs["state_dict"], t_delayed = SparseAutoModel._loadable_state_dict(teacher_path)
teacher = AutoModelForSequenceClassification.from_pretrained(teacher_path,**teacher_kwargs,)
```

## **Step 5: Tokenize Dataset**

Run the tokenizer on the dataset. This is standard Hugging Face functionality.


```python
MAX_LEN = 128
def preprocess_fn(examples):
  args = None
  if INPUT_COL_2 is None:
    args = (examples[INPUT_COL_1], )
  else:
    args = (examples[INPUT_COL_1], examples[INPUT_COL_2])
  result = tokenizer(*args,
                   padding="max_length",
                   max_length=min(tokenizer.model_max_length, MAX_LEN),
                   truncation=True)
  return result

tokenized_dataset = dataset_from_json.map(
    preprocess_fn,
    batched=True,
    desc="Running tokenizer on dataset"
)
```

## **Step 6: Run Training**

SparseML has a custom `Trainer` class that inherits from the [Hugging Face `Trainer` Class](https://huggingface.co/docs/transformers/main_classes/trainer). As such, the SparseML `Trainer` has all of the existing functionality of the HF trainer. However, in addition, we can supply a `recipe` and (optionally) a `teacher`.


As we saw above, the `recipe` encodes the sparsity related algorithms and hyperparameters of the training process in a YAML file. The SparseML `Trainer` parses the `recipe` and adjusts the training workflow to apply the algorithms in the recipe.

The `teacher` is an optional argument that instructs SparseML to apply model distillation to support the training process. Here, we pass the `teacher` model with downloaded from the Hugging Face hub.


```python
training_args = TrainingArguments(
    output_dir="./training_output",
    do_train=True,
    do_eval=True,
    resume_from_checkpoint=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    fp16=True)

trainer = Trainer(
    model=model,
    model_state_path=model_path,
    recipe=recipe_path,
    teacher=teacher,
    metadata_args=["per_device_train_batch_size","per_device_eval_batch_size","fp16"],
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics)
```


```python
train_result = trainer.train(resume_from_checkpoint=False)
trainer.save_model()  # Saves the tokenizer too for easy upload
trainer.save_state()
trainer.save_optimizer_and_scheduler(training_args.output_dir)
```

## **Step 7: Export To ONNX**

Run the following to export the model to ONNX. The script creates a `deployment` folder containing ONNX file and the necessary configuration files (e.g. `tokenizer.json`) for deployment with DeepSparse.


```python
!sparseml.transformers.export_onnx \
  --model_path training_output \
  --task text_classification
```


```python
!ls -lh deployment/
```

## Deploy with DeepSparse

Checkout the [DeepSparse repository for more details](https://github.com/neuralmagic/deepsparse/blob/3601e25e95e91930c45129e2087fda926c64247c/docs/use-cases/nlp/text-classification.md) on deploying your sparse models with GPU class performance on CPUs!


```python
!pip install "deepsparse[transformers]==1.7"
```


```python
from deepsparse import Pipeline

# set up pipeline
path = "deployment/"
sentiment_analysis_pipeline = Pipeline.create(
  task="sentiment-analysis",    # name of the task
  model_path=path,              # zoo stub or path to local onnx file
)

# run inference (input is a sentence, output is the prediction)
prediction = sentiment_analysis_pipeline("These Godzilla sequels are definitely overdone at this point.")
print(prediction)
```


```python
prediction = sentiment_analysis_pipeline("I loved that new monster movie!")
print(prediction)
```

For more detailed documentation on how to deploy sentiment analysis models with DeepSparse, [check out the documentation](https://github.com/neuralmagic/deepsparse/blob/3601e25e95e91930c45129e2087fda926c64247c/docs/use-cases/nlp/sentiment-analysis.md).

**Optional:** You can also upload the exported model to HuggingFace Hub for later use with DeepSparse!


```python
!huggingface-cli upload mgoin/bert-base-rotten-tomatoes-pruned90-quant deployment/
```


```python

```
