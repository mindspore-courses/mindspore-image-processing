'''计算混淆矩阵'''
import os
import json

import mindspore
import mindspore.dataset as ds
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from model import MobileNetV2


class ConfusionMatrix():
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    mindspore.context.set_context(device_target="GPU")

    data_transform = ds.transforms.Compose([ds.vision.Resize(256),
                                            ds.vision.CenterCrop(224),
                                            ds.vision.ToTensor(),
                                            ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = os.path.abspath(os.path.join(
        os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set",
                              "flower_data")  # flower data set path
    assert os.path.exists(
        image_path), f"data path {image_path} does not exist."

    validate_dataset = ds.ImageFolderDataset(dataset_dir=image_path + "val")
    validate_dataset = validate_dataset.map(
        operations=data_transform, input_columns=["image"])

    batch_size = 16
    validate_loader = ds.GeneratorDataset(validate_dataset,
                                          shuffle=False,
                                          num_parallel_workers=2)
    validate_loader = validate_loader.batch(batch_size=batch_size)

    net = MobileNetV2(num_classes=5)
    # load pretrain weights
    model_weight_path = "./MobileNetV2.ckpt"
    assert os.path.exists(
        model_weight_path), "cannot find {} file".format(model_weight_path)
    param_not_load, _ = mindspore.load_param_into_net(
        net, mindspore.load_checkpoint(model_weight_path))
    print(param_not_load)

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(
        json_label_path), f"cannot find {json_label_path} file"
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=5, labels=labels)
    net.set_train(False)
    for val_data in tqdm(validate_loader):
        val_images, val_labels = val_data
        outputs = net(val_images)
        outputs = mindspore.ops.softmax(outputs, axis=1)
        outputs = mindspore.ops.argmax(outputs, dim=1)
        confusion.update(outputs.numpy(),
                         val_labels.numpy())
    confusion.plot()
    confusion.summary()
