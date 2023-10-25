'''train'''
# pylint: disable=E0401
import os
import datetime

import mindspore as ms
from mindspore import dataset, nn, Tensor

import transforms
from my_dataset import VOCDataSet
from src import SSD300, Backbone
import train_utils.train_eval_utils as utils
from plot_curve import plot_loss_and_lr, plot_map


def create_model(num_classes=21):
    '''create model'''
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    # pre_train_path = "./src/resnet50.pth"
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)

    # https://ngc.nvidia.com/catalog/models -> search ssd -> download FP32
    pre_ssd_path = "./src/nvidia_ssdpyt_fp32.ckpt"
    if os.path.exists(pre_ssd_path) is False:
        raise FileNotFoundError(
            "nvidia_ssdpyt_fp32.pt not find in {}".format(pre_ssd_path))
    pre_weights_dict = ms.load_checkpoint(pre_ssd_path)

    # 删除类别预测器权重，注意，回归预测器的权重可以重用，因为不涉及num_classes
    del_conf_loc_dict = {}
    for k, v in pre_weights_dict.items():
        split_key = k.split(".")
        if "conf" in split_key:
            continue
        del_conf_loc_dict.update({k: v})

    missing_keys, unexpected_keys = ms.load_param_into_net(model,
                                                           del_conf_loc_dict, strict_load=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    return model


def main(parser_data):
    '''main'''
    ms.set_context(device_target="GPU")

    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    results_file = "results{}.txt".format(
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transforms.Compose([transforms.SSDCropping(),
                                     transforms.Resize(),
                                     transforms.ColorJitter(),
                                     transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.Normalization(),
                                     transforms.AssignGTtoDefaultBox()]),
        "val": transforms.Compose([transforms.Resize(),
                                   transforms.ToTensor(),
                                   transforms.Normalization()])
    }

    VOC_root = parser_data.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError(
            "VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = VOCDataSet(
        VOC_root, "2012", data_transform['train'], train_set='train.txt')
    # 注意训练时，batch_size必须大于1
    batch_size = parser_data.batch_size
    assert batch_size > 1, "batch size must be greater than 1"

    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % nw)
    train_data_loader = dataset.GeneratorDataset(train_dataset,
                                                 shuffle=True,
                                                 num_parallel_workers=nw)
    train_data_loader = train_data_loader.batch(
        batch_size=batch_size, per_batch_map=train_dataset.collate_fn)

    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(
        VOC_root, "2012", data_transform['val'], train_set='val.txt')
    val_data_set_loader = dataset.GeneratorDataset(val_dataset,
                                                   shuffle=False,
                                                   num_parallel_workers=nw)
    val_data_set_loader = val_data_set_loader.batch(
        batch_size=batch_size, per_batch_map=val_dataset.collate_fn)

    model = create_model(num_classes=parser_data.num_classes+1)

    # define optimizer
    lr = Tensor(utils.get_lr(global_step=0 * batch_size,
                             lr_init=parser_data.lr, lr_end=parser_data.lr * 0.05, lr_max=0.05,
                             warmup_epochs=2, total_epochs=parser_data.epochs-parser_data.start_epoch, steps_per_epoch=batch_size))
    params = [p for p in model.get_parameters() if p.requires_grad]
    optimizer = nn.SGD(params,
                       learning_rate=lr,
                       momentum=parser_data.momentum,
                       weight_decay=parser_data.weight_decay)

    if parser_data.resume != "":
        checkpoint = ms.load_checkpoint(parser_data.resume)
        ms.load_param_into_net(model, checkpoint)
        print("the training process from epoch{}...".format(
            parser_data.start_epoch))

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
