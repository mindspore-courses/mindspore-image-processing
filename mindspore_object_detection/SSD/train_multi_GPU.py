'''单机多卡'''
# pylint: disable=E0401
import os

import time
import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.communication import get_rank, get_group_size
from mindspore.communication import init

from my_dataset import create_dataset
from src import SSD300Vgg16, GeneratDefaultBoxes, init_net_param, get_lr, class_loss
from train_utils import ssd_eval, SsdInferWithDecoder
from plot_curve import plot_loss_and_lr


def main(parser_data):
    '''主函数'''
    ms.set_context(device_target="GPU")
    init("nccl")
    ms.set_auto_parallel_context(
        parallel_mode=ms.ParallelMode.AUTO_PARALLEL, gradients_mean=True)

    default_boxes = GeneratDefaultBoxes().default_boxes

    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    rank_id = get_rank()
    rank_size = get_group_size()

    dataset = create_dataset(
        parser_data.data_path, batch_size=5, device_num=rank_size, rank=rank_id, use_multiprocessing=True)
    dataset_size = dataset.get_dataset_size()

    image, get_loc, gt_label, num_matched_boxes = next(
        dataset.create_tuple_iterator())

    # Network definition and initialization
    network = SSD300Vgg16()
    init_net_param(network)

    # Define the learning rate
    lr = Tensor(get_lr(global_step=0 * dataset_size,
                       lr_init=0.001, lr_end=0.001 * 0.05, lr_max=0.05,
                       warmup_epochs=2, total_epochs=60, steps_per_epoch=dataset_size))

    # Define the optimizer
    opt = nn.Momentum(filter(lambda x: x.requires_grad, network.get_parameters()), lr,
                      0.9, 0.00015, float(1024))

    # Define the forward procedure
    def forward_fn(x, gt_loc, gt_label, num_matched_boxes):
        pred_loc, pred_label = network(x)
        mask = ops.less(0, gt_label).astype(ms.float32)
        num_matched_boxes = ops.sum(num_matched_boxes.astype(ms.float32))

        # Positioning loss
        mask_loc = ops.tile(ops.expand_dims(mask, -1), (1, 1, 4))
        smooth_l1 = nn.SmoothL1Loss()(pred_loc, gt_loc) * mask_loc
        loss_loc = ops.sum(ops.sum(smooth_l1, -1), -1)

        # Category loss
        loss_cls = class_loss(pred_label, gt_label)
        loss_cls = ops.sum(loss_cls, (1, 2))

        return ops.sum((loss_cls + loss_loc) / num_matched_boxes)

    grad_fn = ms.value_and_grad(
        forward_fn, None, opt.parameters, has_aux=False)
    # loss_scaler = DynamicLossScaler(1024, 2, 1000)

    # Gradient updates
    def train_step(x, gt_loc, gt_label, num_matched_boxes):
        loss, grads = grad_fn(x, gt_loc, gt_label, num_matched_boxes)
        opt(grads)
        return loss

    print("=================== Starting Training =====================")
    t_loss = []
    t_lr = []
    for epoch in range(parser_data.epochs):
        network.set_train(True)
        begin_time = time.time()
        for _, (image, get_loc, gt_label, num_matched_boxes) in enumerate(dataset.create_tuple_iterator()):
            loss = train_step(image, get_loc, gt_label, num_matched_boxes)
        end_time = time.time()
        times = end_time - begin_time
        print(f"Epoch:[{int(epoch + 1)}/{int(60)}], "
              f"loss:{loss} , "
              f"time:{times}s ")
        t_loss.append(loss)
        t_lr = lr
    ms.save_checkpoint(network, "ssd-60_9.ckpt")
    plot_loss_and_lr(t_loss, t_lr)
    print("=================== Training Success =====================")
    ds = create_dataset(parser_data.data_test_path, batch_size=parser_data.batch_size,
                        is_training=False, use_multiprocessing=False)
    net = SsdInferWithDecoder(network, Tensor(default_boxes), 'ssd-60_9.ckpt')
    ssd_eval(ds, net,  parser_data.anno_json)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练文件
    parser.add_argument('--anno-json', default='./', help='anno-json')

    # 训练数据集的根目录
    parser.add_argument('--data-path', default='./', help='dataset')
    parser.add_argument('--data-test-path', default='./', help='dataset')
    # 文件保存地址
    parser.add_argument(
        '--output-dir', default='./save_weights', help='path where to save')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练的batch size
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='batch size when training.')

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
