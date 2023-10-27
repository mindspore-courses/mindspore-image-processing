'''模型训练'''
# pylint:disable = E0401
import os
import argparse

import mindspore
import mindspore.dataset as ds
from mindspore.train.summary import SummaryRecord

from my_dataset import MyDataSet
from model import swin_tiny_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    '''主函数'''
    mindspore.set_context(device_target="GPU")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
        args.data_path)

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

    model = create_model(num_classes=args.num_classes)

    if args.weights != "":
        assert os.path.exists(
            args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = mindspore.load_checkpoint(args.weights)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        mindspore.load_param_into_net(model, weights_dict)

    if args.freeze_layers:
        for name, para in model.get_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad = False
            else:
                print(f"training {name}")

    pg = [p for p in model.get_parameters() if p.requires_grad]
    optimizer = mindspore.nn.AdamWeightDecay(
        pg, learning_rate=args.lr, weight_decay=1E-2)

    with SummaryRecord(log_dir="./summary_dir", network=model) as summary_record:
        for epoch in range(args.epochs):
            # train
            train_loss, train_acc = train_one_epoch(model=model,
                                                    optimizer=optimizer,
                                                    data_loader=train_loader,
                                                    epoch=epoch)

            # validate
            val_loss, val_acc = evaluate(model=model,
                                         data_loader=val_loader,
                                         epoch=epoch)

            tags = ["train_loss", "train_acc",
                    "val_loss", "val_acc", "learning_rate"]
            summary_record.add_value('scalar', tags[0], train_loss)
            summary_record.add_value('scalar', tags[1], train_acc)
            summary_record.add_value('scalar', tags[2], val_loss)
            summary_record.add_value('scalar', tags[3], val_acc)
            summary_record.add_value(
                'scalar', tags[4], optimizer.learning_rate.data.asnumpy())

            mindspore.save_checkpoint(model,
                                      f"./weights/model-{epoch}.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.ckpt',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
