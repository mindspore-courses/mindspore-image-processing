'''train utils'''
# pylint: disable=E0401, E1120
import math

import numpy as np
from mindspore import value_and_grad, ops, nn

from eval_utils import F1Score, MeanAbsoluteError


def get_params_groups(model: nn.Cell, weight_decay: float = 1e-4):
    '''get_params_groups'''
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.cells_and_names():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group


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


def criterion(inputs, target):
    '''loss'''
    losses = [ops.binary_cross_entropy_with_logits(
        inputs[i], target) for i in range(len(inputs))]
    total_loss = sum(losses)

    return total_loss


def train_one_epoch(model, optimizer, data_loader):
    '''train one epoch'''
    model.set_train(True)

    # Define forward function

    def forward_fn(net, data, targets):
        output = net(data)
        loss = criterion(output, targets)
        return loss

    # Get gradient function
    grad_fn = value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(net, data, targets):
        loss, grads = grad_fn(net, data, targets)
        optimizer(grads)
        return loss

    losses = 0
    i = 0
    for i, [images, targets] in enumerate(data_loader):
        loss = train_step(model, images, targets)

        losses += loss
    losses /= (i+1)
    return losses, optimizer.learning_rate.data.asnumpy()


def evaluate(model, data_loader):
    '''test model'''
    model.set_train(False)
    mae_metric = MeanAbsoluteError()
    f1_metric = F1Score()

    for image, targets in data_loader:
        output = model(image)
        mae_metric.update(output, targets)
        f1_metric.update(output, targets)

    return mae_metric, f1_metric
