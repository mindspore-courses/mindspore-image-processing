"""
该脚本能够把验证集中预测错误的图片挑选出来，并记录在record.txt中
"""
import os
import json
import argparse
import sys

import mindspore
import mindspore.dataset as ds
from tqdm import tqdm

from my_dataset import MyDataSet
from model import swin_base_patch4_window12_384_in22k as create_model
from utils import read_split_data


def main(args):
    '''主函数'''
    mindspore.context.set_context(device_target="GPU")

    _, _, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 384
    data_transform = {
        "val": ds.transforms.Compose([ds.vision.Resize(int(img_size * 1.143)),
                                      ds.vision.CenterCrop(img_size),
                                      ds.vision.ToTensor(),
                                      ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    val_loader = ds.GeneratorDataset(val_dataset,
                                     shuffle=False,
                                     num_parallel_workers=nw)
    val_loader = val_loader.batch(
        batch_size=batch_size, per_batch_map=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes)

    assert os.path.exists(
        args.weights), f"cannot find {args.weights} file"
    param_not_load, _ = mindspore.load_param_into_net(
        model, mindspore.load_checkpoint(args.weights))
    print(param_not_load)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(
        json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r", encoding='utf-8')
    class_indict = json.load(json_file)

    model.set_train(False)
    with open("record.txt", "w", encoding='utf-8') as f:
        # validate
        data_loader = tqdm(val_loader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images, labels = data
            pred = model(images)
            pred_classes = mindspore.ops.max(pred, axis=1)[1]
            contrast = mindspore.ops.equal(pred_classes, labels).tolist()
            labels = labels.tolist()
            pred_classes = pred_classes.tolist()
            for i, flag in enumerate(contrast):
                if flag is False:
                    file_name = val_images_path[batch_size * step + i]
                    true_label = class_indict[str(labels[i])]
                    false_label = class_indict[str(pred_classes[i])]
                    f.write(
                        f"{file_name}  TrueLabel:{true_label}  PredictLabel:{false_label}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=2)

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="/data/flower_photos")

    # 训练权重路径
    parser.add_argument('--weights', type=str, default='./weights/model-19.ckpt',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--device', default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
