'''train utils'''
# pylint: disable=E0401
import math
import time

import numpy as np

from train_utils import KpLoss, EvalCOCOMetric
import transforms

from mindspore import ops, value_and_grad


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

    mse = KpLoss()

    # Define forward function
    def forward_fn(net, loss, data, label):
        results = net(data)
        losses = loss(results, label)
        return losses, results

    # Get gradient function
    grad_fn = value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(net, loss, data, label):
        (losses, logits), grads = grad_fn(net, loss, data, label)
        optimizer(grads)
        return losses, logits

    for _, [images, targets] in enumerate(data_loader):
        images = ops.stack([image for image in images])
        losses, _ = train_step(model, mse, images, targets)

    return losses, optimizer.learning_rate.data.asnumpy()


def evaluate(model, data_loader, flip=False, flip_pairs=None):
    '''test model'''
    if flip:
        assert flip_pairs is not None, "enable flip must provide flip_pairs."

    model.set_train(False)
    key_metric = EvalCOCOMetric(
        data_loader.dataset.coco, "keypoints", "key_results.json")
    for image, targets in data_loader:
        images = ops.stack([img for img in image])

        model_time = time.time()
        outputs = model(images)

        if flip:
            flipped_images = transforms.flip_images(images)
            flipped_outputs = model(flipped_images)
            flipped_outputs = transforms.flip_back(flipped_outputs, flip_pairs)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            outputs = (outputs + flipped_outputs) * 0.5

        model_time = time.time() - model_time

        # decode keypoint
        reverse_trans = [t["reverse_trans"] for t in targets]
        outputs = transforms.get_final_preds(
            outputs, reverse_trans, post_processing=True)

        key_metric.update(targets, outputs)

    coco_info = key_metric.evaluate()

    return coco_info
