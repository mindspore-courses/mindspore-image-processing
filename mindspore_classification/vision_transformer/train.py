import os
import math
import argparse

import mindspore
import mindspore.dataset as ds
from mindspore.train.summary import SummaryRecord


from my_dataset import MyDataSet
from vit_model import vit_b_16_384 as create_model
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    '''主函数'''
    mindspore.set_context(device_target="GPU")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
        args.data_path)

    data_transform = {
        "train": ds.transforms.Compose([ds.vision.RandomResizedCrop(224),
                                        ds.vision.RandomHorizontalFlip(),
                                        ds.vision.ToTensor(),
                                        ds.vision.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": ds.transforms.Compose([ds.vision.Resize(256),
                                      ds.vision.CenterCrop(224),
                                      ds.vision.ToTensor(),
                                      ds.vision.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

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
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = ds.GeneratorDataset(
        train_dataset, ["data", "label"], num_parallel_workers=nw, shuffle=True)
    train_loader = train_loader.batch(
        batch_size=batch_size, per_batch_map=train_dataset.collate_fn)

    val_loader = ds.GeneratorDataset(
        val_dataset, ["data", "label"], num_parallel_workers=nw, shuffle=False)
    val_loader = val_loader.batch(
        batch_size=batch_size, per_batch_map=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes,
                         has_logits=False)

    if args.weights != "":
        assert os.path.exists(
            args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = mindspore.load_checkpoint(args.weights)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(mindspore.load_param_into_net(model, weights_dict))

    if args.freeze_layers:
        for name, para in model.trainable_params():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad = False
            else:
                print("training {}".format(name))

    pg = [p for p in model.get_parameters() if p.requires_grad]

    decay_steps = 4
    natural_exp_decay_lr = mindspore.nn.NaturalExpDecayLR(
        args.lr, args.lrf, decay_steps, True)
    optimizer = mindspore.nn.SGD(
        pg, learning_rate=natural_exp_decay_lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf

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
                    "val_loss", "val_acc", "lr"]
            summary_record.add_value('scalar', tags[0], train_loss)
            summary_record.add_value('scalar', tags[1], train_acc)
            summary_record.add_value('scalar', tags[2], val_loss)
            summary_record.add_value('scalar', tags[3], val_acc)
            summary_record.add_value(
                'scalar', tags[4], optimizer.learning_rate.data.asnumpy())

            mindspore.save_checkpoint(
                model, "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.ckpt',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
