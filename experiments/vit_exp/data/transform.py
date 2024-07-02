import torchvision


def get_transform(train, input_size, mean=[0, 0, 0], std=[1., 1., 1.], eval_crop_ratio=1., train_aug=None):
    if train:
        transforms_train_ls = [
            torchvision.transforms.Resize(input_size, interpolation=3),
            torchvision.transforms.RandomCrop(input_size, padding=4, padding_mode='reflect'),
            torchvision.transforms.RandomHorizontalFlip()]

        # if aug_cfg is not None and aug_cfg.METHOD:
        #     aug_module = importlib.import_module(aug_cfg.MODULE)
        #     aug_method = getattr(aug_module, aug_cfg.METHOD)
        #     transforms_train_ls += [aug_method(**aug_cfg.ARGS)]
        if train_aug is not None:
            transforms_train_ls += [train_aug]

        transforms_train_ls += [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)
        ]

        transforms = torchvision.transforms.Compose(transforms_train_ls)
    else:
        size = int(input_size / eval_crop_ratio)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size, interpolation=3),
            torchvision.transforms.CenterCrop(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)
        ])
    return transforms