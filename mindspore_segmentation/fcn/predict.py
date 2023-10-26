'''预测'''
# pylint: disable=E0401
import os
import json
import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import dataset as ds


from src import fcn_resnet50


def main():
    '''main'''
    ms.set_context(device_target="GPU")

    aux = False  # inference time not need aux_classifier
    classes = 20
    weights_path = "./save_weights/model_29.ckpt"
    img_path = "./test.jpg"
    palette_path = "./palette.json"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # create model
    model = fcn_resnet50(aux=aux, num_classes=classes+1)

    # delete weights about aux_classifier
    weights_dict = ms.load_checkpoint(weights_path)
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    ms.load_param_into_net(model, weights_dict)

    # load image
    original_img = Image.open(img_path)

    # from pil image to tensor and normalize
    data_transform = ds.transforms.Compose([ds.vision.Resize(520),
                                            ds.vision.ToTensor(),
                                            ds.vision.Normalize(mean=(0.485, 0.456, 0.406),
                                                                std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    # expand batch dimension
    img = ms.ops.unsqueeze(img, dim=0)

    model.set_train(False)  # 进入验证模式

    # init model
    img_height, img_width = img.shape[-2:]
    init_img = ms.ops.zeros((1, 3, img_height, img_width))
    model(init_img)

    output = model(img)

    prediction = output['out'].argmax(1).squeeze(0)
    prediction = prediction.asnumpy().astype(np.uint8)
    mask = Image.fromarray(prediction)
    mask.putpalette(pallette)
    mask.save("test_result.png")


if __name__ == '__main__':
    main()
