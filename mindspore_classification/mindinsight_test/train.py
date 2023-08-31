'''模型训练'''
# pylint:disable=E0401
import os
import math
import argparse

import mindspore
import mindspore.dataset as ds
from mindspore.train.summary import SummaryRecord

from model import resnet34
from my_dataset import MyDataSet
from data_utils import read_split_data, plot_class_preds
from train_eval_utils import train_one_epoch, evaluate

# mindinsight start --summary-base-dir ./summary_dir


def main(args):
    '''主函数'''
    mindspore.set_context(device_target="GPU")

    print(args)
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 划分数据为训练集和验证集
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
        args.data_path)

    # 定义训练以及预测时的预处理方法
    img_size = 224
    data_transform = {
        "train": ds.transforms.Compose([ds.vision.RandomResizedCrop(img_size),
                                        ds.vision.RandomHorizontalFlip(),
                                        ds.vision.ToTensor(),
                                        ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": ds.transforms.Compose([ds.vision.Resize(int(img_size * 1.143)),
                                      ds.vision.CenterCrop(img_size),
                                      ds.vision.ToTensor(),
                                      ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    # 实例化验证数据集
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])

    batch_size = args.batch_size
    # 计算使用num_workers的数量
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = ds.GeneratorDataset(
        train_data_set, ["data", "label"], num_parallel_workers=nw, shuffle=True)
    train_loader = train_loader.batch(
        batch_size=batch_size, per_batch_map=train_data_set.collate_fn)

    val_loader = ds.GeneratorDataset(
        val_data_set, ["data", "label"], num_parallel_workers=nw, shuffle=False)
    val_loader = val_loader.batch(
        batch_size=batch_size, per_batch_map=val_data_set.collate_fn)

    # 实例化模型
    model = resnet34(num_classes=args.num_classes)

    # 如果存在预训练权重则载入
    if os.path.exists(args.weights):
        weights_dict = mindspore.load_checkpoint(args.weights)
        mindspore.load_param_into_net(model, weights_dict)
    else:
        print("not using pretrain-weights.")

    # 是否冻结权重
    if args.freeze_layers:
        print("freeze layers except fc layer.")
        for name, para in model.trainable_params():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad = False

    pg = [p for p in model.get_parameters() if p.requires_grad]
    # 动态学习率

    class MyLR(mindspore.nn.LearningRateSchedule):
        '''自定义动态学习率'''

        def __init__(self, lr, lrf, epochs):
            super().__init__()
            self.lr = lr
            self.lrf = lrf
            self.epochs = epochs

        def construct(self, global_step):
            '''学习率计算'''
            return ((1 + math.cos(global_step * math.pi / self.epochs)) / 2) * \
                (1 - self.lrf) + self.lrf + self.lr  # cosine

    lr = MyLR(args.lr, args.lrf, args.epochs)
    optimizer = mindspore.nn.SGD(
        pg, learning_rate=lr, momentum=0.9, weight_decay=0.005)

    with SummaryRecord(log_dir="./summary_dir", network=model) as summary_record:
        for epoch in range(args.epochs):
            # train
            mean_loss = train_one_epoch(model=model,
                                        optimizer=optimizer,
                                        data_loader=train_loader,
                                        epoch=epoch)

            # validate
            acc = evaluate(model=model,
                           data_loader=val_loader)

            # add loss, acc and lr
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["train_loss", "accuracy", "learning_rate"]
            summary_record.add_value('scalar', tags[0], mean_loss)
            summary_record.add_value('scalar', tags[1], acc)
            summary_record.add_value(
                'scalar', tags[2], optimizer.learning_rate.data.asnumpy())

            # add conv1 weights
            summary_record.add_value('histogram', 'conv1', model.conv1.weight)
            summary_record.add_value('histogram', 'layer1/block0/conv1', model.layer1[0].conv1.weight)

            # save weights
             mindspore.save_checkpoint(model,
                       "./weights/model-{}.ckpt".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    img_root = "/home/wz/my_project/my_github/data_set/flower_data/flower_photos"
    parser.add_argument('--data-path', type=str, default=img_root)

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='resNet34.ckpt',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
