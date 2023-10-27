'''模型训练'''
# pylint: disable=E0401
import os
import sys
import json

import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
from tqdm import tqdm

from model_v2 import MobileNetV2


def main():
    '''主函数'''
    mindspore.set_context(device_target="GPU")

    batch_size = 16
    epochs = 5

    data_transform = {
        "train": ds.transforms.Compose([ds.vision.RandomResizedCrop(224),
                                        ds.vision.RandomHorizontalFlip(),
                                        ds.vision.ToTensor(),
                                        ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": ds.transforms.Compose([ds.vision.Resize(256),
                                      ds.vision.CenterCrop(224),
                                      ds.vision.ToTensor(),
                                      ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(
        os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set",
                              "flower_data")  # flower data set path
    assert os.path.exists(
        image_path), f"{image_path} path does not exist."

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = {'daisy': 0, 'dandelion': 1,
                   'roses': 2, 'sunflower': 3, 'tulips': 4}
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_str)

    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    train_dataset = ds.ImageFolderDataset(os.path.join(
        image_path, "train"), shuffle=True, num_parallel_workers=nw)
    train_dataset = train_dataset.map(
        operations=data_transform["train"], input_columns="image")

    train_num = len(train_dataset)

    train_loader = train_dataset.batch(
        batch_size=batch_size, drop_remainder=True)
    train_loader = train_loader.create_dict_iterator()

    validate_dataset = ds.ImageFolderDataset(os.path.join(
        image_path, "val"), shuffle=False, num_parallel_workers=nw)
    validate_dataset = validate_dataset.map(
        operations=data_transform["val"], input_columns="image")

    val_num = len(validate_dataset)

    validate_loader = validate_dataset.batch(
        batch_size=4, drop_remainder=True)
    validate_loader = validate_loader.create_dict_iterator()

    print(
        f"using {train_num} images for training, {val_num} images for validation.")

    # create model
    net = MobileNetV2(num_classes=5)

    # load pretrain weights
    model_weight_path = "./mobilenet_v2.ckpt"
    assert os.path.exists(
        model_weight_path), f"file {model_weight_path} dose not exist."
    pre_weights = mindspore.load_checkpoint(model_weight_path)

    # delete classifier weights
    for k in list(pre_weights.keys()):
        if "head" in k:
            del pre_weights[k]
    mindspore.load_param_into_net(net, pre_weights)

    # freeze features weights
    for param in net.features.get_parameters():
        param.requires_grad = False

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.get_parameters() if p.requires_grad]
    optimizer = nn.Adam(params, learning_rate=0.0001)

    best_acc = 0.0
    save_path = './MobileNetV2.ckpt'
    train_steps = len(train_loader)

    # 前向传播
    def forward_fn(data, label):
        logits = net(data)
        loss = loss_function(logits, label)
        return loss, logits

    # 梯度函数
    grad_fn = mindspore.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True)

    # 更新，训练
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        optimizer(grads)
        return loss, logits

    for epoch in range(epochs):
        # train
        net.set_train(True)
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for _, data in enumerate(train_bar):
            images, labels = data

            loss, _ = train_step(images, labels)

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.set_train(False)
        acc = 0.0  # accumulate accurate number / epoch

        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images)
            # loss = loss_function(outputs, test_labels)
            predict_y = mindspore.ops.max(outputs, axis=1)[1]
            acc += mindspore.ops.equal(predict_y, val_labels).sum().item()

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                       epochs)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            mindspore.save_checkpoint(net, save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
