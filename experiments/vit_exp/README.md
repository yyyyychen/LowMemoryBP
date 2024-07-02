# ViT Experiments

This subdirectory contains instructions to fine-tune pretrained ViT on image classification tasks.

## Package Requirements

1. Make sure that `torch` and `lomem` is installed (according to the [guidance](../../README.md))
2. Make sure that `yacs` is installed (`pip install yacs`)
3. Make sure that `timm` is installed (`pip install timm`)

## Datasets Preparation

In the ViT experiments, we use CIFAR (`cifar10` and `cifar100`) and FGVC (`CUB-200-2011`, `NABirds`, `Oxford Flowers`, `Stanford Dogs` and `Stanford Cars`).

We suggest putting these datasets in the `./datasets` subfolder, otherwise you should modify the `DATA.PATH` in the config files.

- To download CIFAR, run the following python script:
```python
import torchvision

datasets_path = './datasets'
torchvision.datasets.CIFAR10(root=datasets_path, download=True)
torchvision.datasets.CIFAR100(root=datasets_path, download=True)
```

- To download FGVC, follow the [VPT](https://github.com/KMnP/vpt) repository.

After download these dataset, your datasets directory should have the following file structure:

```
.
├── cifar-100-python
│   ├── file.txt~
│   ├── meta
│   ├── test
│   └── train
├── cifar-10-batches-py
│   ├── batches.meta
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── readme.html
│   └── test_batch
└── FGVC
    ├── CUB_200_2011
    ├── nabirds
    ├── OxfordFlower
    ├── README.txt
    ├── Stanford_cars
    └── Stanford_dogs
```

## Download Pretrained Checkpoints

We suggest putting the pretrained checkpoints in the `./ckpt/` subfolder, otherwise you should modify the `MODEL.CHECKPOINT.PATH` in the config files.

|  Backbone  | Pretrained Dataset | Link |
|  --------  | -----------------  | ---- |
| ViT-B/16   | ImageNet21k        | https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz |
| ViT-L/16   | ImageNet21k        | https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz |

## Fine-tuning Commands

We put fine-tuning configure files in the `./configs` subfolder.
You can run any configure by the following command:
```python
python main.py --config-file config_path

# for example
python main.py --config-file configs/cifar100/lora_all_ViT_B_regelu2_msln.yaml
```