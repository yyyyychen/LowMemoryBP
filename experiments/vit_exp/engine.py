import time
import torch
import utils
import warnings

def train_one_epoch(model : torch.nn.Module, criterion, optimizer, data_loader, device, epoch, cfg, scaler):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")

    for i in range(len(optimizer.param_groups)):
        metric_logger.add_meter(f"lr{i}", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    max_mem = 0
    total_imgs = 0

    total_start_time = time.time()
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, cfg.LOG.PRINT_FREQ, header)):
        start_time = time.time()
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)

        if i % cfg.TRAIN.GRAD_ACCUM == 0:
            torch.cuda.reset_peak_memory_stats()

        if cfg.TRAIN.DISTRIBUTED:
            if (i + 1) % cfg.TRAIN.GRAD_ACCUM != 0:
                with model.no_sync():
                    with torch.cuda.amp.autocast(enabled=cfg.TRAIN.AMP):
                        output = model(image)
                        loss = criterion(output, target) / cfg.TRAIN.GRAD_ACCUM
                    scaler.scale(loss).backward()
            else:
                with torch.cuda.amp.autocast(enabled=cfg.TRAIN.AMP):
                    output = model(image)
                    loss = criterion(output, target) / cfg.TRAIN.GRAD_ACCUM
                scaler.scale(loss).backward()
        else:
            with torch.cuda.amp.autocast(enabled=cfg.TRAIN.AMP):
                output = model(image)
                loss = criterion(output, target) / cfg.TRAIN.GRAD_ACCUM

            scaler.scale(loss).backward()


        if (i + 1) % cfg.TRAIN.GRAD_ACCUM == 0:
            if cfg.TRAIN.CLIP_GRAD_NORM is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.TRAIN.CLIP_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        lr_ls = {f'lr{i}': optimizer.param_groups[i]["lr"] for i in range(len(optimizer.param_groups))}
        metric_logger.update(loss=loss.item(), **lr_ls)
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        total_imgs += image.size(0)
        max_mem = max(torch.cuda.max_memory_allocated() / 1024 ** 2, max_mem)

        time.sleep(0.001)

    total_time = time.time() - total_start_time

    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, max_mem, total_imgs/total_time


def evaluate(model : torch.nn.Module, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    max_mem = 0
    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            torch.cuda.reset_peak_memory_stats()
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

            num_processed_samples += batch_size
            max_mem = max(torch.cuda.max_memory_allocated() / 1024 ** 2, max_mem)
    # gather the stats from all processes

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg, max_mem