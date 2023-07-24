---
license: apache-2.0
base_model: distilbert-base-uncased
tags:
- generated_from_trainer
datasets:
- swag
metrics:
- accuracy
model-index:
- name: distilbert-swag
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilbert-swag

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the swag dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9079
- Accuracy: 0.7105

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.9186        | 1.0   | 2298 | 0.7658          | 0.6950   |
| 0.5843        | 2.0   | 4597 | 0.7453          | 0.7059   |
| 0.3548        | 3.0   | 6894 | 0.9079          | 0.7105   |


### Framework versions

- Transformers 4.31.0
- Pytorch 2.0.1
- Datasets 2.13.1
- Tokenizers 0.13.3
