import numpy as np


class AnchorGenerator():
    '''Generator'''

    def __init__(self, base_size, scales, ratios):
        self.base_size = base_size
        self.scales = np.array(scales)
        self.ratios = np.array(ratios)
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        '''生成feather map中一个点的anchors'''
        w = self.base_size
        h = self.base_size
        x_ctr = 0.5 * (w - 1)
        y_ctr = 0.5 * (h - 1)
        h_ratios = np.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        ws = (w * w_ratios[:, None] * self.scales[None, :]).reshape(-1)
        hs = (h * h_ratios[:, None] * self.scales[None, :]).reshape(-1)
        base_anchors = np.stack([
            x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
        ], axis=-1).round()
        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = np.repeat(x.reshape(1, len(x)), len(y), axis=0).reshape(-1)
        yy = np.repeat(y, len(x))
        if row_major:
            return xx, yy
        return yy, xx

    def grid_anchors(self, featmap_size, stride=16):
        '''根据feature map的大小，生成对应的所有anchors'''
        base_anchors = self.base_anchors

        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride
        shift_y = np.arange(0, feat_h) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = np.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)
        shifts = shifts.astype(base_anchors.dtype)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.reshape(-1, 4)

        return all_anchors
