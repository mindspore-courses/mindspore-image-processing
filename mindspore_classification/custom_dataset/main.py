'''主程序'''
# pylint:disable=E0401, W0611
import os

import mindspore
import mindspore.dataset as ds

from my_dataset import MyDataSet
from utils import read_split_data, plot_data_loader_image

# https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
root = "/home/wz/my_github/data_set/flower_data/flower_photos"  # 数据集所在根目录


def main():
    '''主函数'''
    mindspore.set_context(device_target="GPU")

    train_images_path, train_images_label, _, _ = read_split_data(
        root)

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

    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    batch_size = 8
    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers')
    train_loader = ds.GeneratorDataset(
        train_data_set, ["data", "label"], num_parallel_workers=nw, shuffle=True)
    train_loader = train_loader.batch(
        batch_size=batch_size, per_batch_map=train_data_set.collate_fn)

    # plot_data_loader_image(train_loader)

    for _, data in enumerate(train_loader):
        images, labels = data
        print(images)
        print(labels)


if __name__ == '__main__':
    main()
