'''模型验证'''
import os
import json

import mindspore
from PIL import Image
import mindspore.dataset as ds
import matplotlib.pyplot as plt

from model import convnext_tiny as create_model


def main():
    '''主函数'''
    mindspore.set_context(device_target="GPU")

    num_classes = 5
    img_size = 224
    data_transform = ds.transforms.Compose(
        [ds.vision.Resize(int(img_size * 1.14)),
         ds.vision.CenterCrop(img_size),
         ds.vision.ToTensor(),
         ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

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
    model = create_model(num_classes=num_classes)
    # load model weights
    model_weight_path = "./weights/best_model.ckpt"
    param_dict = mindspore.load_checkpoint(model_weight_path)
    param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
    print(param_not_load)
    model.set_train(False)
    # predict class
    output = mindspore.ops.squeeze(model(img))
    predict = mindspore.ops.softmax(output, axis=0)
    predict_cla = mindspore.ops.argmax(predict).asnumpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
