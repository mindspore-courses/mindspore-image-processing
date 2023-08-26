'''模型预测'''
# pylint:disable=E0401, W0611
import os
import json

import mindspore
import mindspore.dataset as ds
from PIL import Image
import matplotlib.pyplot as plt

from vit_model import vit_b_16_384 as create_model


def main():
    '''主函数'''
    mindspore.set_context(device_target="GPU")

    data_transform = ds.transforms.Compose(
        [ds.vision.Resize(256),
         ds.vision.CenterCrop(224),
         ds.vision.ToTensor(),
         ds.vision.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(
        img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = mindspore.ops.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(
        json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=5, has_logits=False)
    # load model weights
    model_weight_path = "./weights/model-9.ckpt"
    param_dict = mindspore.load_checkpoint(model_weight_path)
    param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
    print(param_not_load)
    model.set_train(False)

    # predict class
    output = mindspore.ops.squeeze(model(img))
    predict = mindspore.ops.softmax(output, axis=0)
    predict_cla = mindspore.ops.argmax(predict).asnumpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].asnumpy())
    plt.title(print_res)
    for i, _ in enumerate(predict):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].asnumpy()))
    plt.show()


if __name__ == '__main__':
    main()
