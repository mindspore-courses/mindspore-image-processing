"""
refer to:
https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html
"""
# pylint: disable = E0401
import os
import argparse

from absl import logging
from tqdm import tqdm
import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore_gs import SimulatedQuantizationAwareTraining as SimQAT
from mindcv.models.regnet import resnet34 as create_model

from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate

logging.set_verbosity(logging.FATAL)


def export_onnx(model, onnx_filename, onnx_bs):
    '''导出onnx'''
    model.set_train(False)

    print(f"Export ONNX file: {onnx_filename}")
    dummy_input = mindspore.ops.randn(onnx_bs, 3, 224, 224)
    mindspore.export(net=model,                       # model being run
                     # model input (or a tuple for multiple inputs)
                     inputs=dummy_input,
                     # where to save the model (can be a file or file-like object)
                     file_name=onnx_filename,
                     file_format='ONNX')


def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    algo = SimQAT()
    model = algo.convert(model)

    for i, (images, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(images)
        if i >= num_batches:
            break

    # Disable calibrators
    model = algo.apply(model)


def main(args):
    '''量化训练主函数'''
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
    print(f'Using {nw} dataloader workers every process')

    train_loader = ds.GeneratorDataset(train_dataset,
                                       shuffle=True,
                                       num_parallel_workers=nw)
    train_loader = train_loader.apply(train_dataset.collate_fn)
    train_loader = train_loader.batch(batch_size=batch_size)

    val_loader = ds.GeneratorDataset(val_dataset,
                                     shuffle=False,
                                     num_parallel_workers=nw)
    val_loader = val_loader.apply(val_dataset.collate_fn)
    val_loader = val_loader.batch(batch_size=batch_size)

    model = create_model(num_classes=args.num_classes)
    assert os.path.exists(
        args.weights), f"weights file: '{args.weights}' not exist."
    param_not_load, _ = mindspore.load_param_into_net(
        model, mindspore.load_checkpoint(args.weights))
    print(param_not_load)

    # It is a bit slow since we collect histograms on CPU
    collect_stats(model, val_loader, num_batches=1000 // batch_size)
    # validate
    evaluate(model=model, data_loader=val_loader, epoch=0)

    mindspore.save_checkpoint(model, "quant_model_calibrated.ckpt")

    if args.qat:
        # Quantization Aware Training #
        pg = [p for p in model.trainable_params() if p.requires_grad]

        def dynamic_lr(lr, total_step, step_per_epoch):
            # Scheduler(half of a cosine period)
            lrs = []
            for i in range(total_step):
                current_epoch = i // step_per_epoch
                factor = current_epoch // 5
                lrs.append(lr * factor)
            return lrs

        decay_lr = dynamic_lr(lr=args.lr, total_step=200, step_per_epoch=10)

        optimizer = nn.SGD(pg, learning_rate=decay_lr,
                           momentum=0.9, weight_decay=5E-5)

        for epoch in range(args.epochs):
            # train
            train_one_epoch(model=model, optimizer=optimizer,
                            data_loader=train_loader, epoch=epoch)

            # validate
            evaluate(model=model, data_loader=val_loader, epoch=epoch)

    export_onnx(model, args.onnx_filename, args.onnx_bs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")

    # 训练好的权重路径
    parser.add_argument('--weights', type=str, default='./resNet(flower).pth',
                        help='trained weights path')

    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument(
        '--onnx-filename', default='resnet34.onnx', help='save onnx model filename')
    parser.add_argument('--onnx-bs', default=1,
                        help='save onnx model batch size')
    parser.add_argument('--qat', type=bool, default=True,
                        help='whether use quantization aware training')

    opt = parser.parse_args()

    main(opt)
