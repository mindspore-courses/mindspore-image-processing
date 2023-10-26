'''单机多卡，数据并行'''
# pylint: disable=E0401
import os
import time
import argparse
import datetime
from typing import Union, List

import mindspore as ms
from mindspore import Tensor, dataset, nn
from mindspore.communication import get_rank, get_group_size, init

from src import u2net_full
from train_utils import train_one_epoch, evaluate, get_params_groups, get_lr
from my_dataset import DUTSDataset
import transforms as T


class SODPresetTrain:
    '''SODPresetTrain'''

    def __init__(self, base_size: Union[int, List[int]], crop_size: int,
                 hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=True),
            T.RandomCrop(crop_size),
            T.RandomHorizontalFlip(hflip_prob),
            T.Normalize(mean=mean, std=std)
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class SODPresetEval:
    '''SODPresetEval'''

    def __init__(self, base_size: Union[int, List[int]], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(base_size, resize_mask=False),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def main(args):
    '''main'''
    ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
    init("nccl")
    ms.set_auto_parallel_context(
        parallel_mode=ms.ParallelMode.AUTO_PARALLEL, gradients_mean=True)

    batch_size = args.batch_size

    # get rank_id and rank_size
    rank_id = get_rank()
    rank_size = get_group_size()

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = DUTSDataset(
        args.data_path, train=True, transforms=SODPresetTrain([320, 320], crop_size=288))
    val_dataset = DUTSDataset(
        args.data_path, train=False, transforms=SODPresetEval([320, 320]))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_data_loader = dataset.GeneratorDataset(train_dataset,
                                                 shuffle=True,
                                                 num_shards=rank_size,
                                                 shard_id=rank_id,
                                                 num_parallel_workers=num_workers)
    train_data_loader = train_data_loader.batch(
        batch_size=batch_size, per_batch_map=train_dataset.collate_fn)

    val_data_loader = dataset.GeneratorDataset(val_dataset,
                                               shuffle=False,
                                               num_shards=rank_size,
                                               shard_id=rank_id,
                                               num_parallel_workers=num_workers)
    val_data_loader = val_data_loader.batch(
        batch_size=1, per_batch_map=val_dataset.collate_fn)

    model = u2net_full()

    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    # define optimizer
    lr = Tensor(get_lr(global_step=0 * batch_size,
                       lr_init=args.lr, lr_end=args.lr * 0.05, lr_max=0.05,
                       warmup_epochs=2, total_epochs=args.epochs-args.start_epoch, steps_per_epoch=batch_size))
    optimizer = nn.AdamWeightDecay(
        params_group, learning_rate=lr, weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = ms.load_checkpoint(args.resume)
        ms.load_param_into_net(model, checkpoint)

    current_mae, current_f1 = 1.0, 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader)

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间
            mae_metric, f1_metric = evaluate(model, val_data_loader)
            mae_info, f1_info = mae_metric.compute(), f1_metric.compute()
            print(
                f"[epoch: {epoch}] val_MAE: {mae_info:.3f} val_maxF1: {f1_info:.3f}")
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                write_info = f"[epoch: {epoch}] train_loss: {mean_loss:.4f} lr: {lr:.6f} " \
                             f"MAE: {mae_info:.3f} maxF1: {f1_info:.3f} \n"
                f.write(write_info)

            # save_best
            if current_mae >= mae_info and current_f1 <= f1_info:
                ms.save_checkpoint(model, "save_weights/model_best.ckpt")
        # only save latest 10 epoch weights
        if os.path.exists(f"save_weights/model_{epoch-10}.ckpt"):
            os.remove(f"save_weights/model_{epoch-10}.ckpt")

        ms.save_checkpoint(model, f"save_weights/model_{epoch}.ckpt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    '''config'''
    parser = argparse.ArgumentParser(description="mindspore u2net training")

    parser.add_argument("--data-path", default="./", help="DUTS root")
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--epochs", default=360, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--eval-interval", default=10,
                        type=int, help="validation interval default 10 Epochs")

    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    return parser.parse_args()


if __name__ == '__main__':

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(parse_args())
