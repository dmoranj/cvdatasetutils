#########################################################################
#
# Taken from pytorch/vision/references/detection
#
#########################################################################
import math
import sys
import time
import torch
import time
from apex import amp
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, test=False,
                    max_iters=9, max_oom_errors=5, error_delay=5, max_nan_errors=5,
                    batch_accumulator=1, history_freq=3, half_precision=False):
    if test:
        model.train()
    else:
        model.train()
        lr_scheduler = None

        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    oom_error = 0
    nan_error = 0
    counter = 0

    history = []

    count_accumulator = 0

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        try:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) / batch_accumulator

            if count_accumulator == batch_accumulator:
                apply_gradient = True
                count_accumulator = 1
            else:
                apply_gradient = False
                count_accumulator += 1

            if not test:
                if apply_gradient:
                    optimizer.zero_grad()

                if half_precision:
                    with amp.scale_loss(losses, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    losses.backward()

                if apply_gradient:
                    optimizer.step()

                    if lr_scheduler is not None:
                        lr_scheduler.step()


            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced = {key: value.detach().cpu() for key, value in loss_dict_reduced.items()}
            losses_reduced = sum(loss.double() for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                if nan_error > max_nan_errors:
                    print("Loss is {}, stopping training".format(loss_value))
                    print(loss_dict_reduced)
                    sys.exit(1)
                else:
                    print("Nan error found. Waiting for stochasticity to work the miracle.")
                    continue

            metric_logger.update(loss=losses_reduced.detach(), **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if counter % history_freq == 0:
                history.append([
                    counter, loss_value, *[float(value.numpy()) for key, value in loss_dict_reduced.items()]
                ])

            oom_error = 0

            counter += 1

            if max_iters is not None and counter > max_iters:
                break

        except RuntimeError as e:
                if 'out of memory' in str(e) and oom_error < max_oom_errors:
                    print('| WARNING: ran out of memory, retrying batch on 5s')
                    clean_from_error(error_delay, model, oom_error)
                    log_error(e)
                    oom_error += 1
                elif 'illegal' in str(e) and oom_error < max_oom_errors:
                    print('Illegal memory access found, retrying anyway, and logging')

                    clean_from_error(error_delay, model, oom_error)
                    log_error(e)
                    oom_error += 1
                else:
                    print('| Couldnt recover from OOM error or similar', e)
                    raise e

    return history


def clean_from_error(error_delay, model, oom_error):
    time.sleep(error_delay * (2 ** oom_error))
    for p in model.parameters():
        if p.grad is not None:
            del p.grad

    torch.cuda.empty_cache()


def log_error(e):
    with open('./memoryErrors.log', 'a') as f:
        f.write(str(e))
        f.write('\n')
        f.close()


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, max_iters=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    iters = 0

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        if max_iters is not None:
            if iters > max_iters:
                break
            else:
                iters += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
