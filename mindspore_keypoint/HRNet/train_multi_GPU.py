'''单机多卡训练，数据并行'''
# pylint: disable=E0401, W0621
import json
import os
import datetime
import numpy as np

import mindspore
from mindspore import dataset, nn, Tensor
from mindspore.communication import get_rank, get_group_size, init

import transforms
from model import HighResolutionNet
from my_dataset_coco import CocoKeypoint
from train_utils import train_eval_utils as utils

from plot_curve import plot_loss_and_lr, plot_map


def create_model(num_joints, load_pretrain_weights=True):
    '''creat model'''
    model = HighResolutionNet(base_channel=32, num_joints=num_joints)

    if load_pretrain_weights:
        # 载入预训练模型权重
        weights_dict = mindspore.load_checkpoint("./hrnet_w32.ckpt")

        for k in list(weights_dict.keys()):
            # 如果载入的是imagenet权重，就删除无用权重
            if ("head" in k) or ("fc" in k):
                del weights_dict[k]

            # 如果载入的是coco权重，对比下num_joints，如果不相等就删除
            if "final_layer" in k:
                if weights_dict[k].shape[0] != num_joints:
                    del weights_dict[k]

        missing_keys, _ = mindspore.load_param_into_net(model,
                                                        weights_dict, strict_load=False)
        if len(missing_keys) != 0:
            print("missing_keys: ", missing_keys)

    return model


def main(args):
    '''main'''
    mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="GPU")
    init("nccl")
    mindspore.set_auto_parallel_context(
        parallel_mode=mindspore.ParallelMode.AUTO_PARALLEL, gradients_mean=True)

    # 用来保存coco_info的文件
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    key_results_file = f"results{now}.txt"

    with open(args.keypoints_path, "r", encoding='utf-8') as f:
        person_kps_info = json.load(f)

    fixed_size = args.fixed_size
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)
    kps_weights = np.array(person_kps_info["kps_weights"],
                           dtype=np.float32).reshape((args.num_joints,))
    data_transform = {
        "train": transforms.Compose([
            transforms.HalfBody(
                0.3, person_kps_info["upper_body_ids"], person_kps_info["lower_body_ids"]),
            transforms.AffineTransform(
                scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            transforms.RandomHorizontalFlip(
                0.5, person_kps_info["flip_pairs"]),
            transforms.KeypointToHeatMap(
                heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(
                scale=(1.25, 1.25), fixed_size=fixed_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
    }

    data_root = args.data_path

    # load train data set
    # coco2017 -> annotations -> person_keypoints_train2017.json
    train_dataset = CocoKeypoint(
        data_root, "train", transforms=data_transform["train"], fixed_size=args.fixed_size)

    # load validation data set
    # coco2017 -> annotations -> person_keypoints_val2017.json
    val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], fixed_size=args.fixed_size,
                               det_json_path=args.person_det)

    print("Creating data loaders")
    batch_size = args.batch_size
    # get rank_id and rank_size
    rank_id = get_rank()
    rank_size = get_group_size()

    train_data_loader = dataset.GeneratorDataset(train_dataset,
                                                 shuffle=True,
                                                 num_shards=rank_size,
                                                 shard_id=rank_id,
                                                 num_parallel_workers=args.workers)
    train_data_loader = train_data_loader.batch(
        batch_size=batch_size, per_batch_map=train_dataset.collate_fn)

    # load validation data set
    # coco2017 -> annotations -> person_keypoints_val2017.json
    val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], fixed_size=args.fixed_size,
                               det_json_path=args.person_det)
    val_data_loader = dataset.GeneratorDataset(val_dataset,
                                               shuffle=False,
                                               num_shards=rank_size,
                                               shard_id=rank_id,
                                               num_parallel_workers=args.workers)
    val_data_loader = val_data_loader.batch(
        batch_size=batch_size, per_batch_map=val_dataset.collate_fn)

    print("Creating model")
    # create model num_classes equal background + classes
    model = create_model(num_joints=args.num_joints)

    # define optimizer
    lr = Tensor(utils.get_lr(global_step=0 * batch_size,
                             lr_init=args.lr, lr_end=args.lr * 0.05, lr_max=0.05,
                             warmup_epochs=2, total_epochs=args.epochs-args.start_epoch, steps_per_epoch=batch_size))
    params = [p for p in model.get_parameters() if p.requires_grad]
    optimizer = nn.AdamWeightDecay(params,
                                   learning_rate=lr,
                                   weight_decay=args.wd)

    if args.resume != "":
        checkpoint = mindspore.load_checkpoint(args.resume)
        mindspore.load_param_into_net(model, checkpoint)
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    print("Start training")
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(
            model, optimizer, train_data_loader)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_loader,
                                   flip=True, flip_pairs=person_kps_info["flip_pairs"])

        # write into txt
        with open(key_results_file, "a", encoding='utf-8') as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info +
                           [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # @0.5 mAP

        # save weights
        mindspore.save_checkpoint(
            model, "./save_weights/model-{}.ckpt".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练文件的根目录(coco2017)
    parser.add_argument(
        '--data-path', default='/data/coco2017', help='dataset')
    # COCO数据集人体关键点信息
    parser.add_argument('--keypoints-path', default="./person_keypoints.json", type=str,
                        help='person_keypoints.json path')
    # 原项目提供的验证集person检测信息，如果要使用GT信息，直接将该参数置为None，建议设置成None
    parser.add_argument('--person-det', type=str, default=None)
    parser.add_argument(
        '--fixed-size', default=[256, 192], nargs='+', type=int, help='input size')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-joints', default=17, type=int,
                        help='num_joints(num_keypoints)')
    # 每块GPU上的batch_size
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start-epoch', default=0,
                        type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=210, type=int, metavar='N',
                        help='number of total epochs to run')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 学习率
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate, 0.001 is the default value for training '
                             'on 4 gpus and 32 images_per_gpu')
    # AdamW的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 文件保存地址
    parser.add_argument(
        '--output-dir', default='./multi_train', help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        os.mkdir(args.output_dir)

    main(args)
