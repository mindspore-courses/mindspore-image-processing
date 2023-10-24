'''train model'''
# pylint: disable=E0401, W0621, W0611
import os
import datetime

import mindspore as ms
from mindspore import dataset, nn, Tensor

import transforms
from network_files import FasterRCNN, AnchorsGenerator, FastRCNNPredictor
from backbone import MobileNetV2, vgg, resnet50_fpn_backbone
from my_dataset import VOCDataSet
from train_utils import train_eval_utils as utils

from plot_curve import plot_loss_and_lr, plot_map


def create_model(num_classes, load_pretrain_weights=True):
    '''create model'''
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    backbone = resnet50_fpn_backbone(pretrain_path="./backbone/resnet50.ckpt",
                                     norm_layer=nn.BatchNorm2d,
                                     trainable_layers=3)
    # backbone = MobileNetV2(weights_path="./backbone/mobilenet_v2.ckpt").features
    # backbone.out_channels = 1280  # 设置对应backbone输出特征矩阵的channels
    # vgg_feature = vgg(model_name="vgg16", weights_path="./backbone/vgg16.ckpt").features
    # backbone = nn.SequentialCell(*list(vgg_feature._modules.values())[:-1])  # 删除features中最后一个Maxpool层
    # backbone.out_channels = 512
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = FasterRCNN(backbone=backbone, num_classes=91)

    if load_pretrain_weights:
        # 载入预训练模型权重
        weights_dict = ms.load_checkpoint(
            "./backbone/fasterrcnn_resnet50_fpn_coco.ckpt")
        missing_keys, unexpected_keys = ms.load_param_into_net(model,
                                                               weights_dict, strict_load=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main(args):
    '''main'''
    ms.set_context(device_target="GPU")

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = args.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError(
            f"VOCdevkit dose not in path:'{VOC_root}'.")

    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = VOCDataSet(
        VOC_root, "2012", data_transform["train"], "train.txt")

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % nw)

    train_data_loader = dataset.GeneratorDataset(train_dataset,
                                                 shuffle=True,
                                                 num_parallel_workers=nw)
    train_data_loader = train_data_loader.batch(
        batch_size=batch_size, per_batch_map=train_dataset.collate_fn)

    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(
        VOC_root, "2012", data_transform["val"], "val.txt")
    val_data_set_loader = dataset.GeneratorDataset(val_dataset,
                                                   shuffle=False,
                                                   num_parallel_workers=nw)
    val_data_set_loader = val_data_set_loader.batch(
        batch_size=1, per_batch_map=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(num_classes=args.num_classes + 1)

    # define optimizer
    lr = Tensor(utils.get_lr(global_step=0 * batch_size,
                             lr_init=args.lr, lr_end=args.lr * 0.05, lr_max=0.05,
                             warmup_epochs=2, total_epochs=args.epochs-args.start_epoch, steps_per_epoch=batch_size))
    params = [p for p in model.get_parameters() if p.requires_grad]
    optimizer = nn.SGD(params,
                       learning_rate=lr,
                       momentum=args.momentum,
                       weight_decay=args.weight_decay)

    if args.resume != "":
        checkpoint = ms.load_checkpoint(args.resume)
        ms.load_param_into_net(model, checkpoint)
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(
            model, optimizer, train_data_loader)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # evaluate on the test dataset
        coco_info = utils.evaluate(model, val_data_set_loader)

        # write into txt
        with open(results_file, "a", encoding='utf-8') as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info +
                           [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # @0.5 mAP

        # save weights
        ms.save_checkpoint(
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

    # 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='./', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=20,
                        type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument(
        '--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str,
                        help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0,
                        type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch_size', default=8, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)