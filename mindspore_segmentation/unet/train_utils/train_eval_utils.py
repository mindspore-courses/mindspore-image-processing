'''train utils'''
# pylint: disable=E0401
import math

import numpy as np
from mindspore import value_and_grad, ops, Tensor

from .dice_coefficient_loss import dice_loss, build_target
import train_utils.eval_utils as utils


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


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    '''loss'''
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        loss = ops.cross_entropy(
            x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(x, dice_target, multiclass=True,
                              ignore_index=ignore_index)
        losses[name] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def train_one_epoch(model, optimizer, data_loader, num_classes):
    '''train one epoch'''
    model.set_train(True)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = Tensor([1.0, 2.0])
    else:
        loss_weight = None

    # Define forward function

    def forward_fn(net, data, targets, num_classes):
        output = net(data)
        loss = criterion(output, targets, loss_weight,
                         num_classes=num_classes, ignore_index=255)
        return loss

    # Get gradient function
    grad_fn = value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(net, data, targets, num_classes):
        loss, grads = grad_fn(net, data, targets, num_classes)
        optimizer(grads)
        return loss

    losses = 0
    i = 0
    for i, [images, targets] in enumerate(data_loader):
        loss = train_step(model, images, targets, num_classes)

        losses += loss
    losses /= (i+1)
    return losses, optimizer.learning_rate.data.asnumpy()


def evaluate(model, data_loader, num_classes):
    '''test model'''
    model.set_train(False)
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)

    for image, targets in data_loader:
        images = list(img for img in image)
        output = model(images)
        output = output['out']
        confmat.update(targets.flatten(), output.argmax(1).flatten())
        dice.update(output, targets)

    return confmat, dice.value.item()
