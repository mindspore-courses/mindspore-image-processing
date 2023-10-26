'''单机多卡，数据并行'''
# pylint: disable=E0401
import os
import argparse
import time
import datetime
import mindspore as ms
from mindspore import dataset, nn, Tensor
from mindspore.communication import get_rank, get_group_size, init
from src import deeplabv3_resnet50
from train_utils import train_eval_utils as utils
from my_dataset import VOCSegmentation
import transforms as T


class SegmentationPresetTrain:
    '''SegmentationPresetTrain'''

    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
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

    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train):
    '''transform'''
    base_size = 520
    crop_size = 480

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)


def create_model(aux, num_classes, pretrain=True):
    '''model'''
    model = deeplabv3_resnet50(aux=aux, num_classes=num_classes)

    if pretrain:
        weights_dict = ms.load_checkpoint("./deeplabv3_resnet50_coco.ckpt")

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = ms.load_param_into_net(
            model, weights_dict, strict_load=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


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

    # segmentation nun_classes + background
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息
    results_file = "results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    train_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt")

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt_name="val.txt")

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = dataset.GeneratorDataset(train_dataset,
                                            shuffle=True,
                                            num_shards=rank_size,
                                            shard_id=rank_id,
                                            num_parallel_workers=num_workers)
    train_loader = train_loader.batch(
        batch_size=batch_size, per_batch_map=train_dataset.collate_fn)

    val_loader = dataset.GeneratorDataset(train_dataset,
                                          shuffle=False,
                                          num_shards=rank_size,
                                          shard_id=rank_id,
                                          num_parallel_workers=num_workers)
    val_loader = val_loader.batch(
        batch_size=1, per_batch_map=val_dataset.collate_fn)

    model = create_model(aux=args.aux, num_classes=num_classes)

    params_to_optimize = [
        {"params": [p for p in model.backbone.trainable_params()
                    if p.requires_grad]},
        {"params": [p for p in model.classifier.trainable_params()
                    if p.requires_grad]}
    ]

    if args.aux:
        params = [p for p in model.aux_classifier.trainable_params()
                  if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    # define optimizer
    lr = Tensor(utils.get_lr(global_step=0 * batch_size,
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

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(
            model, optimizer, train_loader)

        # evaluate on the test dataset
        confmat = utils.evaluate(model, val_loader, num_classes)

        val_info = str(confmat)
        print(val_info)
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        # save weights
        ms.save_checkpoint(
            model, "./save_weights/model-{}.ckpt".format(epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    '''config'''
    parser = argparse.ArgumentParser(
        description="mindspore deeplabv3 training")

    parser.add_argument("--data-path", default="/data/", help="VOCdevkit root")
    parser.add_argument("--num-classes", default=20, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10,
                        type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    return parser.parse_args()


if __name__ == '__main__':

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(parse_args())
