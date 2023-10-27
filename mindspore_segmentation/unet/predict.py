'''预测'''
# pylint: disable=E0401
import os

import minsdpore as ms
from mindspore import dataset, ops

import numpy as np
from PIL import Image

from src import UNet


def main():
    '''main'''
    classes = 1  # exclude background
    weights_path = "./save_weights/best_model.ckpt"
    img_path = "./DRIVE/test/images/01_test.tif"
    roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    ms.set_context(device_target="GPU")

    # create model
    model = UNet(in_channels=3, num_classes=classes+1, base_c=32)

    # load weights
    ms.load_param_into_net(model, ms.load_checkpoint(weights_path))

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    data_transform = dataset.transforms.Compose([dataset.vision.ToTensor(),
                                                 dataset.vision.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = ops.unsqueeze(img, dim=0)

    model.set_train(False)  # 进入验证模式
    # init model
    img_height, img_width = img.shape[-2:]
    init_img = ops.zeros((1, 3, img_height, img_width))
    model(init_img)

    output = model(img)

    prediction = output['out'].argmax(1).squeeze(0)
    prediction = prediction.to("cpu").numpy().astype(np.uint8)
    # 将前景对应的像素值改成255(白色)
    prediction[prediction == 1] = 255
    # 将不敢兴趣的区域像素设置成0(黑色)
    prediction[roi_img == 0] = 0
    mask = Image.fromarray(prediction)
    mask.save("test_result.png")


if __name__ == '__main__':
    main()
