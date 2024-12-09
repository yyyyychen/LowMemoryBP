# Experiments for Llama 7b/13b

This subdirectory contains instructions to fine-tune pretrained Llama 7b/13b on alpaca dataset.

## Package Requirements

1. Make sure that `torch` and `lomem` is installed (according to the [guidance](../../README.md))
2. Make sure that `transformers` is installed (`pip install transformers`)
3. Make sure that `bitsandbytes` is installed (`pip install bitsandbytes`)

## Model Preparation

1. Download Pretrained Checkpoints

We suggest putting the pretrained checkpoints in the `./ckpt/` subfolder, otherwise you should modify the `model_name_or_path` in the shell files.

We recommend you download checkpoints from huggyllama/llama-7b and huggyllama/llama-13b in Huggingface.

|  Backbone   | Link |
|  --------   | ---- |
| Llama-7b    | https://huggingface.co/huggyllama/llama-7b  |
| Llama-13b   | https://huggingface.co/huggyllama/llama-13b |

2. For simplified usage, we recommend you merge LayerNorm before fine-tuning. But you can also merge LayerNorm during fine-tuning.

```{shell}
bash ./scripts/merge_llama_7b.sh
bash ./scripts/merge_llama_13b.sh
```

## Fine-tuning Commands

1. If you **DON'T** use merge LayerNorm

```{shell}
bash ./scripts/finetune_llama_7b.sh
bash ./scripts/finetune_llama_13b.sh
```

`using_method`: you can choose `wo` or `activation`, which means `original fine-tune` or `only use ReGELU2` respectively.

2. If you use merge LayerNorm

```{shell}
bash ./scripts/finetune_llama_7b_mergeln.sh
bash ./scripts/finetune_llama_13b_mergeln.sh
```

`using_method`: you can choose `mergeln` or `activation_and_mergeln`, which means `only use MS-RMSNorm` or `use MS-RMSNorm and ReSiLU2` respectively.