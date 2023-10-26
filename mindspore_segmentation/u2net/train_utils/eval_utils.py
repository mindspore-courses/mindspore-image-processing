'''utils'''
from mindspore import Tensor, ops


class F1Score():
    """
    refer: https://github.com/xuebinqin/DIS/blob/main/IS-Net/basics.py
    """

    def __init__(self, threshold: float = 0.5):
        self.precision_cum = None
        self.recall_cum = None
        self.num_cum = None
        self.threshold = threshold

    def update(self, pred: Tensor, gt: Tensor):
        '''update'''
        batch_size, _, h, w = gt.shape
        assert batch_size == 1, f"validation mode batch_size must be 1, but got batch_size: {batch_size}."
        resize_pred = ops.interpolate(
            pred, (h, w), mode="bilinear", align_corners=False)
        gt_num = ops.sum(ops.gt(gt, self.threshold).float())

        pp = resize_pred[ops.gt(gt, self.threshold)]  # 对应预测map中GT为前景的区域
        nn = resize_pred[ops.le(gt, self.threshold)]  # 对应预测map中GT为背景的区域

        pp_hist = ops.histc(pp, bins=255, min=0.0, max=1.0)
        nn_hist = ops.histc(nn, bins=255, min=0.0, max=1.0)

        # Sort according to the prediction probability from large to small
        pp_hist_flip = ops.flipud(pp_hist)
        nn_hist_flip = ops.flipud(nn_hist)

        pp_hist_flip_cum = ops.cumsum(pp_hist_flip, axis=0)
        nn_hist_flip_cum = ops.cumsum(nn_hist_flip, axis=0)

        precision = pp_hist_flip_cum / \
            (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
        recall = pp_hist_flip_cum / (gt_num + 1e-4)

        if self.precision_cum is None:
            self.precision_cum = ops.full_like(precision, fill_value=0.)

        if self.recall_cum is None:
            self.recall_cum = ops.full_like(recall, fill_value=0.)

        if self.num_cum is None:
            self.num_cum = ops.zeros([1], dtype=gt.dtype)

        self.precision_cum += precision
        self.recall_cum += recall
        self.num_cum += batch_size

    def compute(self):
        '''compute'''
        pre_mean = self.precision_cum / self.num_cum
        rec_mean = self.recall_cum / self.num_cum
        f1_mean = (1 + 0.3) * pre_mean * rec_mean / \
            (0.3 * pre_mean + rec_mean + 1e-8)
        max_f1 = ops.amax(f1_mean).item()
        return max_f1

    def __str__(self):
        max_f1 = self.compute()
        return f'maxF1: {max_f1:.3f}'


class MeanAbsoluteError():
    '''MeanAbsoluteError'''
    def __init__(self):
        self.mae_list = []

    def update(self, pred: Tensor, gt: Tensor):
        '''update'''
        batch_size, _, h, w = gt.shape
        assert batch_size == 1, f"validation mode batch_size must be 1, but got batch_size: {batch_size}."
        resize_pred = ops.interpolate(
            pred, (h, w), mode="bilinear", align_corners=False)
        error_pixels = ops.sum(
            ops.abs(resize_pred - gt), dim=(1, 2, 3)) / (h * w)
        self.mae_list.extend(error_pixels.tolist())

    def compute(self):
        '''compute'''
        mae = sum(self.mae_list) / len(self.mae_list)
        return mae

    def __str__(self):
        mae = self.compute()
        return f'MAE: {mae:.3f}'
