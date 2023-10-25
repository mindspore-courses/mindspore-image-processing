'''train utils'''
# pylint: disable=E0401
import math
import time

import numpy as np

from train_utils import CocoEvaluator

from mindspore import value_and_grad, ops, Tensor


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
        losses_dict = net(data, targets)
        losses = losses_dict["total_losses"]
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
        images = ops.stack(images, axis=0)

        boxes = []
        labels = []
        img_id = []
        for t in targets:
            boxes.append(t['boxes'])
            labels.append(t['labels'])
            img_id.append(t["image_id"])
        targets = {"boxes": ops.stack(boxes, axis=0),
                   "labels": ops.stack(labels, axis=0),
                   "image_id": Tensor(img_id)}
        targets = dict((k, v) for k, v in targets.items())

        losses = train_step(model, images, targets)

    return losses, optimizer.learning_rate.data.asnumpy()


def evaluate(model, data_loader):
    '''test model'''
    model.set_train(False)
    iou_types = _get_iou_types()
    coco_evaluator = CocoEvaluator(data_loader.dataset.coco, iou_types)

    for image, targets in data_loader:
        images = ops.stack(image, axis=0)

        model_time = time.time()
        results = model(images, targets=None)

        outputs = []
        for index, (bboxes_out, labels_out, scores_out) in enumerate(results):
            # 将box的相对坐标信息（0-1）转为绝对值坐标(xmin, ymin, xmax, ymax)
            height_width = targets[index]["height_width"]
            # 还原回原图尺度
            bboxes_out[:, [0, 2]] = bboxes_out[:, [0, 2]] * height_width[1]
            bboxes_out[:, [1, 3]] = bboxes_out[:, [1, 3]] * height_width[0]

            info = {"boxes": bboxes_out,
                    "labels": labels_out,
                    "scores": scores_out}
            outputs.append(info)

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target,
               output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # numpy to list
    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()

    return coco_info


def _get_iou_types():
    iou_types = ["bbox"]
    return iou_types
