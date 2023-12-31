'''工具类'''
import os
import sys
import json
import pickle
import random
import mindspore
from tqdm import tqdm


def read_split_data(root: str, val_rate: float = 0.2):
    '''数据分割'''
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(
        root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(
        root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key)
                          for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print(f"{sum(every_class_num)} images were found in the dataset.")
    print(f"{len(train_images_path)} images for training.")
    print(f"{len(val_images_path)} images for validation.")

    return train_images_path, train_images_label, val_images_path, val_images_label


def write_pickle(list_info: list, file_name: str):
    '''写入'''
    with open(file_name, 'wb', encoding='utf-8') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    '''读取'''
    with open(file_name, 'rb', encoding='utf-8') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, epoch):
    '''一次迭代'''
    model.set_train(True)
    loss_function = mindspore.nn.CrossEntropyLoss()
    accu_loss = mindspore.ops.zeros(1)  # 累计损失
    accu_num = mindspore.ops.zeros(1)   # 累计预测正确的样本数

    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_function(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = mindspore.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        optimizer(grads)
        return loss, logits

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    step = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        loss, pred = train_step(images, labels)
        pred_classes = mindspore.ops.max(pred, axis=1)[1]
        accu_num += mindspore.ops.equal(pred_classes, labels).sum()

        accu_loss += loss.detach()

        data_loader.desc = f"[train epoch {epoch}] loss: {(accu_loss.item() / (step + 1)):.3f}, acc: {(accu_num.item() / sample_num):.3f}"

        if not mindspore.ops.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def evaluate(model, data_loader, epoch):
    '''验证'''
    loss_function = mindspore.nn.CrossEntropyLoss()

    model.set_train(False)

    accu_num = mindspore.ops.zeros(1)   # 累计预测正确的样本数
    accu_loss = mindspore.ops.zeros(1)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    step = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images)
        pred_classes = mindspore.ops.max(pred, axis=1)[1]
        accu_num += mindspore.ops.equal(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss

        data_loader.desc = f"[valid epoch {epoch}] loss: {(accu_loss.item() / (step + 1)):.3f}, acc: {(accu_num.item() / sample_num):.3f}"

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
