'''train'''
# pylint: disable = E0401
import os
import time
import datetime
import argparse

import mindspore as ms
from mindspore import dataset, Tensor, nn

from src import UNet
from train_utils import train_one_epoch, evaluate, get_lr
from my_dataset import DriveDataset
import transforms as T


class SegmentationPresetTrain:
    '''SegmentationPresetTrain'''

    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    '''SegmentationPresetEval'''

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    '''get_transform'''
    base_size = 565
    crop_size = 480

    if train:
        out = SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        out = SegmentationPresetEval(mean=mean, std=std)
    return out


def create_model(num_classes):
    '''model'''
    model = UNet(in_channels=3, num_classes=num_classes, base_c=32)
    return model


def main(args):
    '''main'''
    ms.set_context(device_target="GPU")
    batch_size = args.batch_size
    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # using compute_mean_std.py
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, mean=mean, std=std))

    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, mean=mean, std=std))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = dataset.GeneratorDataset(train_dataset,
                                            shuffle=True,
                                            num_parallel_workers=num_workers)
    train_loader = train_loader.batch(
        batch_size=batch_size, per_batch_map=train_dataset.collate_fn)

    val_loader = dataset.GeneratorDataset(train_dataset,
                                          shuffle=False,
                                          num_parallel_workers=num_workers)
    val_loader = val_loader.batch(
        batch_size=1, per_batch_map=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)

    params_to_optimize = [p for p in model.get_parameters() if p.requires_grad]

    # define optimizer
    lr = Tensor(get_lr(global_step=0 * batch_size,
                       lr_init=args.lr, lr_end=args.lr * 0.05, lr_max=0.05,
                       warmup_epochs=2, total_epochs=args.epochs-args.start_epoch, steps_per_epoch=batch_size))
    optimizer = nn.SGD(params_to_optimize,
                       learning_rate=lr,
                       momentum=args.momentum,
                       weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = ms.load_checkpoint(args.resume)
        ms.load_param_into_net(model, checkpoint)
        print("the training process from epoch{}...".format(args.start_epoch))

    best_dice = 0.
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(
            model, optimizer, train_loader, num_classes)

        confmat, dice = evaluate(
            model, val_loader, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"dice coefficient: {dice:.3f}")
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"dice coefficient: {dice:.3f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        if args.save_best is True:
            ms.save_checkpoint(model, "save_weights/best_model.ckpt")
        else:
            ms.save_checkpoint(model, f"save_weights/model_{epoch}.ckpt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    '''config'''
    parser = argparse.ArgumentParser(description="mindspore unet training")

    parser.add_argument("--data-path", default="./", help="DRIVE root")
    # exclude background
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True,
                        type=bool, help='only save best dice weights')

    return parser.parse_args()


if __name__ == '__main__':

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(parse_args())
