'''自定义损失函数'''
from mindspore import nn, ops


class KpLoss(nn.Cell):
    '''KplLoss'''

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def construct(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = ops.stack([t["heatmap"] for t in targets])
        # [num_kps] -> [B, num_kps]
        kps_weights = ops.stack([t["kps_weights"] for t in targets])

        # [B, num_kps, H, W] -> [B, num_kps]
        loss = self.criterion(logits, heatmaps).mean(axis=[2, 3])
        loss = ops.sum(loss * kps_weights) / bs
        return loss
