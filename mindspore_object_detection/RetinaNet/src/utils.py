import math
import itertools as it
import numpy as np
from mindspore import nn, ops, Tensor
import mindspore as ms
from mindvision.engine.loss.multiboxloss import SoftmaxCrossEntropyWithLogits


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


class RetinaWithLossCell(nn.Cell):
    """
    Retina with loss.
    """

    def __init__(self, multibox_loss):
        super(RetinaWithLossCell, self).__init__()
        self.loc_weight = 0.5
        self.class_weight = 0.2
        self.landm_weight = 0.3
        self.multibox_loss = multibox_loss

    def construct(self, img, loc_t, conf_t, landm_t):
        pred_loc, pre_conf, pre_landm = self.network(img)
        loss_loc, loss_conf, loss_landm = self.multibox_loss(
            pred_loc, loc_t, pre_conf, conf_t, pre_landm, landm_t)
        return loss_loc * self.loc_weight + loss_conf * self.class_weight + loss_landm * self.landm_weight


class MultiBoxLoss(nn.Cell):
    """"
    MultiBoxLoss for detection.
    """

    def __init__(self, num_classes, num_boxes, neg_pre_positive, batch_size):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.neg_pre_positive = neg_pre_positive
        self.notequal = ops.NotEqual()
        self.less = ops.Less()
        self.tile = ops.Tile()
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()
        self.expand_dims = ops.ExpandDims()
        self.smooth_l1_loss = ops.SmoothL1Loss()
        self.cross_entropy = SoftmaxCrossEntropyWithLogits()
        self.maximum = ops.Maximum()
        self.minimum = ops.Minimum()
        self.sort_descend = ops.TopK(True)
        self.sort = ops.TopK(True)
        self.gather = ops.GatherNd()
        self.max = ops.ReduceMax()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.concat = ops.Concat(axis=1)
        self.reduce_sum2 = ops.ReduceSum(keep_dims=True)
        self.idx = Tensor(np.reshape(
            np.arange(batch_size * num_boxes), (-1, 1)), ms.int32)

    def construct(self, loc_data, loc_t, conf_data, conf_t, landm_data, landm_t):
        """Forward pass."""

        # landm loss
        mask_pos1 = ops.cast(
            self.less(0.0, ops.cast(conf_t, ms.float32)), ms.float32)
        reducesumed = self.expand_dims(self.reduce_sum(mask_pos1), 0)
        n1 = self.maximum(reducesumed, 1)
        mask_pos_idx1 = self.tile(self.expand_dims(mask_pos1, -1), (1, 1, 10))
        loss_landm = self.reduce_sum(self.smooth_l1_loss(
            landm_data, landm_t) * mask_pos_idx1)
        loss_landm = loss_landm / n1

        # Localization Loss
        mask_pos = ops.cast(self.notequal(0, conf_t), ms.float32)
        conf_t = ops.cast(mask_pos, ms.int32)

        n = self.maximum(self.expand_dims(self.reduce_sum(mask_pos), 0), 1)
        mask_pos_idx = self.tile(self.expand_dims(mask_pos, -1), (1, 1, 4))
        loss_l = self.reduce_sum(self.smooth_l1_loss(
            loc_data, loc_t) * mask_pos_idx)
        loss_l = loss_l / n

        # Conf Loss
        conf_t_shape = ops.shape(conf_t)
        conf_t = ops.reshape(conf_t, (-1,))
        indices = self.concat((self.idx, ops.reshape(conf_t, (-1, 1))))

        batch_conf = ops.reshape(conf_data, (-1, self.num_classes))
        x_max = self.max(batch_conf)
        loss_c = self.log(self.reduce_sum2(
            self.exp(batch_conf - x_max), 1)) + x_max
        loss_c = loss_c - \
            ops.reshape(self.gather(batch_conf, indices), (-1, 1))
        loss_c = ops.reshape(loss_c, conf_t_shape)

        # hard example mining
        num_matched_boxes = ops.reshape(self.reduce_sum(mask_pos, 1), (-1,))
        neg_masked_cross_entropy = ops.cast(
            loss_c * (1 - mask_pos), ms.float32)

        _, loss_idx = self.sort_descend(
            neg_masked_cross_entropy, self.num_boxes)
        _, relative_position = self.sort(
            ops.cast(loss_idx, ms.float32), self.num_boxes)
        relative_position = ops.cast(relative_position, ms.float32)
        relative_position = relative_position[:, ::-1]
        relative_position = ops.cast(relative_position, ms.int32)

        num_neg_boxes = self.minimum(
            num_matched_boxes * self.neg_pre_positive, self.num_boxes - 1)
        tile_num_neg_boxes = self.tile(self.expand_dims(
            num_neg_boxes, -1), (1, self.num_boxes))
        top_k_neg_mask = ops.cast(
            self.less(relative_position, tile_num_neg_boxes), ms.float32)

        cross_entropy = self.cross_entropy(batch_conf, conf_t)
        cross_entropy = ops.reshape(cross_entropy, conf_t_shape)

        loss_c = self.reduce_sum(
            cross_entropy * self.minimum(mask_pos + top_k_neg_mask, 1))

        loss_c = loss_c / n

        return loss_l, loss_c, loss_landm


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
