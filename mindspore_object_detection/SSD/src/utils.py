import math
import itertools as it
import numpy as np
import mindspore as ms
from mindspore import Tensor, ops


def class_loss(logits, label):
    """Calculate category losses."""
    label = ops.one_hot(label, ops.shape(
        logits)[-1], Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
    weight = ops.ones_like(logits)
    pos_weight = ops.ones_like(logits)
    sigmiod_cross_entropy = ops.binary_cross_entropy_with_logits(
        logits, label, weight.astype(ms.float32), pos_weight.astype(ms.float32))
    sigmoid = ops.sigmoid(logits)
    label = label.astype(ms.float32)
    p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
    modulating_factor = ops.pow(1 - p_t, 2.0)
    alpha_weight_factor = label * 0.75 + (1 - label) * (1 - 0.75)
    focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
    return focal_loss


class GeneratDefaultBoxes():
    """
    Generate Default boxes, follows the order of (W, H, archor_sizes).
    `self.default_boxes` has a shape of [archor_sizes, H, W, 4], the last dimension is [y, x, h, w].
    `self.default_boxes_tlbr` has a shape as `self.default_boxes`, the last dimension is [y1, x1, y2, x2].
    """

    def __init__(self):
        fk = 300 / np.array([8, 16, 32, 64, 100, 300])
        scale_rate = (0.95 - 0.1) / (len([4, 6, 6, 6, 4, 4]) - 1)
        scales = [0.1 + scale_rate *
                  i for i in range(len([4, 6, 6, 6, 4, 4]))] + [1.0]
        self.default_boxes = []
        for idex, feature_size in enumerate([38, 19, 10, 5, 3, 1]):
            sk1 = scales[idex]
            sk2 = scales[idex + 1]
            sk3 = math.sqrt(sk1 * sk2)
            if idex == 0 and not [[2], [2, 3], [2, 3], [2, 3], [2], [2]][idex]:
                w, h = sk1 * math.sqrt(2), sk1 / math.sqrt(2)
                all_sizes = [(0.1, 0.1), (w, h), (h, w)]
            else:
                all_sizes = [(sk1, sk1)]
                for aspect_ratio in [[2], [2, 3], [2, 3], [2, 3], [2], [2]][idex]:
                    w, h = sk1 * math.sqrt(aspect_ratio), sk1 / \
                        math.sqrt(aspect_ratio)
                    all_sizes.append((w, h))
                    all_sizes.append((h, w))
                all_sizes.append((sk3, sk3))

            assert len(all_sizes) == [4, 6, 6, 6, 4, 4][idex]

            for i, j in it.product(range(feature_size), repeat=2):
                for w, h in all_sizes:
                    cx, cy = (j + 0.5) / fk[idex], (i + 0.5) / fk[idex]
                    self.default_boxes.append([cy, cx, h, w])

        def to_tlbr(cy, cx, h, w):
            return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

        # For IoU calculation
        self.default_boxes_tlbr = np.array(
            tuple(to_tlbr(*i) for i in self.default_boxes), dtype='float32')
        self.default_boxes = np.array(self.default_boxes, dtype='float32')


def init_net_param(network, initialize_mode='TruncatedNormal'):
    """Init the parameters in net."""
    params = network.trainable_params()
    for p in params:
        if 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            if initialize_mode == 'TruncatedNormal':
                p.set_data(ms.common.initializer.initializer(ms.common.initializer.TruncatedNormal(
                    0.02), p.data.shape, p.data.dtype))
            else:
                p.set_data(initialize_mode, p.data.shape, p.data.dtype)


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
