# Experiments for RoBERTa:

This subdirectory contains instructions to fine-tune pretrained RoBERTa-base on GLUE benchmark.

## Package Requirements

1. Make sure that `torch` and `lomem` is installed (according to the [guidance](../../README.md))
2. Make sure that `transformers` is installed (`pip install transformers`)

## Model Preparation

1. Download Pretrained Checkpoints

We suggest putting the pretrained checkpoints in the `./ckpt/` subfolder, otherwise you should modify the `model_name_or_path` in the shell files.

We recommend you download checkpoints from FacebookAI/roberta-base in Huggingface.

|         Backbone        | Link |
|         --------        | ---- |
| FacebookAI/roberta-base | https://huggingface.co/FacebookAI/roberta-base  |

2. For simplified usage, we recommend you merge LayerNorm before fine-tuning. But you can also merge LayerNorm during fine-tuning.

```{shell}
bash ./scripts/merge_roberta.sh
```

3. You can download `glue` dataset from nyu-mll/glue in huggingface.

|   Dataset    | Link |
|   -------    | ---- |
| nyu-mll/glue | https://huggingface.co/datasets/nyu-mll/glue |

## Fine-tuning Commands

1. If you **DON'T** use merge LayerNorm

```{shell}
bash ./scripts/finetune_roberta.sh
```

`using_method`: you can choose `wo` or `activation`, which means `original fine-tune` or `only use ReGELU2` respectively.

2. If you use merge LayerNorm

```{shell}
bash ./scripts/finetune_roberta_mergeln.sh
```

`using_method`: you can choose `mergeln` or `activation_and_mergeln`, which means `only use MS-LayerNorm` or `use MS-LayerNorm and ReGELU2` respectively.