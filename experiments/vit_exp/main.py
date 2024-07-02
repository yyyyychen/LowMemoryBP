import datetime
import os
import sys
import time

import torch
import utils
import importlib

from data import get_dataloader
from model import get_model
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from engine import train_one_epoch, evaluate

import utils
import argparse
from config import get_cfg_defaults


def main(cfg):
    device = torch.device(cfg.SYSTEM.DEVICE)


    print("Preparing data")
    data_loader_train, data_loader_test, train_sampler, test_sampler = get_dataloader(
        data_cfg=cfg.DATA, distributed=cfg.TRAIN.DISTRIBUTED)

    print("Creating model")
    num_classes = data_loader_train.dataset._class_num

    model = get_model(cfg.MODEL, num_classes=num_classes, img_size=cfg.DATA.TRANSFORM.IMG_SIZE)

    params_counts = sum(p.numel() for p in model.parameters() if p.requires_grad)

    with open(os.path.join(cfg.LOG.OUTPUT_DIR, "params_counts.txt"), 'w') as f:
        f.write("trainable parameters:\n")
        for n, p in model.named_parameters():
            if p.requires_grad:
                f.write(f"{n}\n")
        f.write(f"the number of trainable parameters: {params_counts}")

    criterion = torch.nn.CrossEntropyLoss()

    lr = cfg.OPTIMIZER.ARGS.lr
    batch_size = cfg.DATA.BATCH_SIZE_PER_GPU * cfg.TRAIN.GRAD_ACCUM
    epochs = cfg.TRAIN.EPOCHS
    # rebuild the optimizer and schedulers
    linear_scaled_lr = lr * batch_size * utils.get_world_size() / 512.0
    cfg.OPTIMIZER.ARGS.lr = linear_scaled_lr

    if utils.is_dist_avail_and_initialized():
        model.to(f'cuda:{cfg.gpu}')
    else:
        model.to('cuda')

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.AMP)

    model_without_ddp = model
    if cfg.TRAIN.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer_v2(model_without_ddp, **cfg.OPTIMIZER.ARGS)

    # model = torch.compile(model)

    # setup learning rate schedule and starting epoch
    updates_per_epoch = len(data_loader_train)
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        num_epochs=epochs,
        updates_per_epoch=updates_per_epoch,
        **cfg.LR_SCHEDULER.ARGS
    )

    start_epoch = 0
    if cfg.TRAIN.RESUME_PATH:
        print(f"Resume from {cfg.TRAIN.RESUME_PATH}")
        checkpoint = torch.load(cfg.TRAIN.RESUME_PATH, map_location="cpu")
        msg = model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        print(msg)
        if not cfg.TRAIN.TEST_ONLY:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1

        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if cfg.TRAIN.TEST_ONLY:
        if cfg.MODEL.CHECKPOINT.FINETUNED_PATH:
            model.load_finetuned_weight(cfg.MODEL.CHECKPOINT.FINETUNED_PATH)
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        evaluate(model, criterion, data_loader_test, device=device)
        return

    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        if cfg.TRAIN.DISTRIBUTED:
            train_sampler.set_epoch(epoch)

        if cfg.MODEL.SCHED.METHOD:
            model_scheduler_module = importlib.import_module(cfg.MODEL.SCHED.MODULE)
            model_scheduler = getattr(model_scheduler_module, cfg.MODEL.SCHED.METHOD)
            model_scheduler(model=model.module, epoch=epoch, total_epoch=epochs, **cfg.MODEL.SCHED.ARGS)

        # set_stepsize(model.module, optimizer.param_groups[0]["lr"])

        loss_train, acc_train, train_max_mem, train_throughput = train_one_epoch(model, criterion, optimizer, data_loader_train, device, epoch, cfg, scaler)
        loss_test, acc_test, test_max_mem = evaluate(model, criterion, data_loader_test, device=device)

        if cfg.LOG.OUTPUT_DIR:
            checkpoint = {
                "model": utils.get_trainable_state_dict(model_without_ddp, ['lora_A', 'lora_B']),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "cfg": cfg,
            }

            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(cfg.LOG.OUTPUT_DIR, "checkpoint.pth"))

            # logging
            if utils.is_main_process():
                log_txt = os.path.join(cfg.LOG.OUTPUT_DIR, 'log.txt')
                content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {loss_test:.5f}, val acc: {acc_test:.5f}, train loss: {loss_train:.5f}, train acc: {acc_train:.5f}, train max mem: {train_max_mem}, test max mem: {test_max_mem}, train throughput: {train_throughput}'
                with open(log_txt, 'a') as appender:
                    appender.write(content + "\n")

        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step(epoch + 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

def default_argument_parser():
    """
    create a simple parser to wrap around config file
    """
    parser = argparse.ArgumentParser(description="visual-finetune")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def setup_experiment():
    args = default_argument_parser().parse_args()

    cfg = get_cfg_defaults()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # print(cfg)

    if cfg.TRAIN.TEST_ONLY:
        return cfg

    # if only test, log setting is skipped

    if cfg.LOG.OUTPUT_DIR:
        utils.mkdir(cfg.LOG.OUTPUT_DIR)

    # check if the exp is already finished
    log_path = os.path.join(cfg.LOG.OUTPUT_DIR, 'log.txt')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            lines = f.readlines()
            if f'Epoch {cfg.TRAIN.EPOCHS - 1}' in lines[-1]:
                sys.exit(0)

    config_save_path = os.path.join(cfg.LOG.OUTPUT_DIR, "config.yaml")
    with open(config_save_path, 'w') as f:
        f.write(cfg.dump())

    if cfg.LOG.OUTPUT_DIR:
        utils.mkdir(cfg.LOG.OUTPUT_DIR)

    if cfg.TRAIN.DISTRIBUTED:
        utils.init_distributed_mode(args)
        cfg.gpu = args.gpu

    if cfg.TRAIN.DETERMINISTIC:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    return cfg


if __name__ == "__main__":
    cfg = setup_experiment()
    main(cfg)