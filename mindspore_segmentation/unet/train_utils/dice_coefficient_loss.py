'''utils'''
import mindspore
from mindspore import ops, Tensor


def build_target(target: Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    on_value, off_value = Tensor(
        1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
    if ignore_index >= 0:
        ignore_mask = ops.equal(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = ops.one_hot(
            dice_target, num_classes, on_value, off_value).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = ops.one_hot(
            dice_target, num_classes, on_value, off_value).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: Tensor, target: Tensor, ignore_index: int = -100, epsilon=1e-6):
    '''dice coeff'''
    # Average of Dice coefficient for all batches, or for a single mask
    # 计算一个batch中所有图片某个类别的dice_coefficient
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            # 找出mask中不为ignore_index的区域
            roi_mask = ops.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = ops.dot(x_i, t_i)
        sets_sum = ops.sum(x_i) + ops.sum(t_i)
        if sets_sum == 0:
            sets_sum = 2 * inter

        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    return d / batch_size


def multiclass_dice_coeff(x: Tensor, target: Tensor, ignore_index: int = -100, epsilon=1e-6):
    """Average of Dice coefficient for all classes"""
    dice = 0.
    for channel in range(x.shape[1]):
        dice += dice_coeff(x[:, channel, ...],
                           target[:, channel, ...], ignore_index, epsilon)

    return dice / x.shape[1]


def dice_loss(x: Tensor, target: Tensor, multiclass: bool = False, ignore_index: int = -100):
    '''Dice loss (objective to minimize) between 0 and 1'''
    x = ops.softmax(x, axis=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)
