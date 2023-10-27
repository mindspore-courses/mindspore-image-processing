'''train utils'''
# pylint: disable=E0401
import math
import time

import numpy as np

from train_utils import EvalCOCOMetric

from mindspore import value_and_grad


def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    """ generate learning rate array"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_end + (lr_max - lr_end) * (1. + math.cos(math.pi *
                                                             (i - warmup_steps) / (total_steps - warmup_steps))) / 2.
        if lr < 0.0:
            lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


def train_one_epoch(model, optimizer, data_loader):
    '''train one epoch'''
    model.set_train(True)

    # Define forward function

    def forward_fn(net, data, targets):
        loss_dict = net(data, targets)
        losses = sum(loss for loss in loss_dict.values())
        return losses

    # Get gradient function
    grad_fn = value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(net, data, targets):
        losses, grads = grad_fn(net, data, targets)
        optimizer(grads)
        return losses

    for _, [images, targets] in enumerate(data_loader):
        images = list(image for image in images)
        targets = [dict(t.items()) for t in targets]
        losses = train_step(model, images, targets)

    return losses, optimizer.learning_rate.data.asnumpy()


def evaluate(model, data_loader):
    '''test model'''
    model.set_train(False)
    det_metric = EvalCOCOMetric(
        data_loader.dataset.coco, iou_type="bbox", results_file_name="det_results.json")
    seg_metric = EvalCOCOMetric(
        data_loader.dataset.coco, iou_type="segm", results_file_name="seg_results.json")

    for image, targets in data_loader:
        images = list(img for img in image)

        model_time = time.time()
        outputs = model(images)

        outputs = [dict(t.items())
                   for t in outputs]
        model_time = time.time() - model_time

        evaluator_time = time.time()
        det_metric.update(targets, outputs)
        seg_metric.update(targets, outputs)
        evaluator_time = time.time() - evaluator_time

    coco_info = det_metric.evaluate()
    seg_info = seg_metric.evaluate()

    return coco_info, seg_info
