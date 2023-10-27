'''模型训练'''
# pylint:disable = E0401
import os
import math
import argparse

import mindspore
import mindspore.dataset as ds
from mindspore.train.summary import SummaryRecord

from model import efficientnet_b0 as create_model
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    '''主函数'''
    mindspore.set_context(device_target="GPU")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
        args.data_path)

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = {
        "train": ds.transforms.Compose([ds.vision.RandomResizedCrop(img_size[num_model]),
                                        ds.vision.RandomHorizontalFlip(),
                                        ds.vision.ToTensor(),
                                        ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": ds.transforms.Compose([ds.vision.Resize(img_size[num_model]),
                                      ds.vision.CenterCrop(
                                          img_size[num_model]),
                                      ds.vision.ToTensor(),
                                      ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')
    train_loader = ds.GeneratorDataset(
        train_dataset, ["data", "label"], num_parallel_workers=nw, shuffle=True)
    train_loader = train_loader.batch(
        batch_size=batch_size, per_batch_map=train_dataset.collate_fn)

    val_loader = ds.GeneratorDataset(
        val_dataset, ["data", "label"], num_parallel_workers=nw, shuffle=False)
    val_loader = val_loader.batch(
        batch_size=batch_size, per_batch_map=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = mindspore.load_checkpoint(args.weights)["model"]
            mindspore.load_param_into_net(model, weights_dict)
        else:
            raise FileNotFoundError(
                "not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.get_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
                para.requires_grad = False
            else:
                print("training {}".format(name))

    pg = [p for _, p in model.get_parameters() if p.requires_grad]
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
        pg, learning_rate=lr, momentum=0.9, weight_decay=1E-4)

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
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            summary_record.add_value('scalar', tags[0], mean_loss)
            summary_record.add_value('scalar', tags[1], acc)
            summary_record.add_value(
                'scalar', tags[2], optimizer.learning_rate.data.asnumpy())

            mindspore.save_checkpoint(model, f"./weights/model-{epoch}.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")

    # download model weights
    parser.add_argument('--weights', type=str, default='./efficientnetb0.ckpt',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
