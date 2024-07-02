import torchvision


datasets_path = './datasets'
torchvision.datasets.CIFAR10(root=datasets_path, download=True)
torchvision.datasets.CIFAR100(root=datasets_path, download=True)