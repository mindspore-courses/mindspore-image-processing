'''模型预测'''
# pylint:disable=E0401, E0611
import os
import json

import mindspore
import mindspore.dataset as ds
from PIL import Image

from model import resnet34


def main():
    '''主函数'''
    mindspore.set_context(device_target="GPU")

    data_transform = ds.transforms.Compose(
        [ds.vision.Resize((256, 256)),
         ds.vision.CenterCrop(224),
         ds.vision.ToTensor(),
         ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = "/data/imgs"
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i)
                     for i in os.listdir(imgs_root) if i.endswith(".jpg")]

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' dose not exist."

    json_file = open(json_path, "r", encoding='utf-8')
    class_indict = json.load(json_file)

    # create model
    model = resnet34(num_classes=5)

    # load model weights
    weights_path = "./resNet34.ckpt"
    assert os.path.exists(
        weights_path), f"file: '{weights_path}' dose not exist."
    param_dict = mindspore.load_checkpoint(weights_path)
    param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
    print(param_not_load)
    model.set_train(False)

    # prediction
    batch_size = 8  # 每次预测时将多少张图片打包成一个batch

    for ids in range(0, len(img_path_list) // batch_size):
        img_list = []
        for img_path in img_path_list[ids * batch_size: (ids + 1) * batch_size]:
            assert os.path.exists(
                    img_path), f"file: '{img_path}' dose not exist."
            img = Image.open(img_path)
            img = data_transform(img)
            img_list.append(img)

        # batch img
        # 将img_list列表中的所有图像打包成一个batch
        batch_img = mindspore.ops.stack(img_list, axis=0)
        # predict class
        output = model(batch_img)
        predict = mindspore.ops.softmax(output, axis=1)
        probs, classes = mindspore.ops.max(predict, axis=1)

         for idx, (pro, cla) in enumerate(zip(probs, classes)):
            print("image: {}  class: {}  prob: {:.3}".format(img_path_list[ids * batch_size + idx],
                                                                 class_indict[str(
                                                                     cla.asnumpy())],
                                                                 pro.asnumpy()))


if __name__ == '__main__':
    main()
