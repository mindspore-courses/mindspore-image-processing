'''模型预测'''
# pylint:disable=E0401, E0611
import os
import json

import mindspore
import mindspore.dataset as ds
from PIL import Image
import matplotlib.pyplot as plt

from model import shufflenet_v2_x1_0


def main():
    '''主函数'''
    mindspore.set_context(device_target="GPU")

    data_transform = ds.transforms.Compose(
        [ds.vision.Resize(256),
         ds.vision.CenterCrop(224),
         ds.vision.ToTensor(),
         ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(
        img_path), f"file: '{img_path}' dose not exist."
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = mindspore.ops.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(
        json_path), f"file: '{json_path}' dose not exist."

    with open(json_path, "r", encoding='utf-8') as f:
        class_indict = json.load(f)

    # create model
    model = shufflenet_v2_x1_0(num_classes=5)
    # load model weights
    model_weight_path = "./weights/model-29.ckpt"
    param_dict = mindspore.load_checkpoint(model_weight_path)
    param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
    print(param_not_load)
    model.set_train(False)

    # predict class
    output = mindspore.ops.squeeze(model(img))
    predict = mindspore.ops.softmax(output, axis=0)
    predict_cla = mindspore.ops.argmax(predict).asnumpy()

    print_res = f"class: {class_indict[str(predict_cla)]}   prob: {predict[predict_cla].asnumpy():.3}"
    plt.title(print_res)
    for i, _ in enumerate(predict):
        print(
            f"class: {class_indict[str(i)]:10}   prob: {predict[i].asnumpy():.3}")
    plt.show()


if __name__ == '__main__':
    main()
