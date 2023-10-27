'''预测'''
# pylint: disable=E0401
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

import mindspore as ms
from mindspore import ops, dataset

from src import u2net_full


def main():
    '''main'''
    weights_path = "./u2net_full.ckpt"
    img_path = "./test.png"
    threshold = 0.5

    assert os.path.exists(img_path), f"image file {img_path} dose not exists."

    ms.set_context(device_target="GPU")

    data_transform = dataset.transforms.Compose([
        dataset.vision.ToTensor(),
        dataset.vision.Resize(320),
        dataset.vision.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
    ])

    origin_img = cv2.cvtColor(cv2.imread(
        img_path, flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    h, w = origin_img.shape[:2]
    img = data_transform(origin_img)
    img = ops.unsqueeze(img, 0)  # [C, H, W] -> [1, C, H, W]

    model = u2net_full()
    ms.load_param_into_net(model, ms.load_checkpoint(weights_path))

    model.set_train(False)

    # init model
    img_height, img_width = img.shape[-2:]
    init_img = ops.zeros((1, 3, img_height, img_width))
    model(init_img)

    pred = model(img)

    pred = ops.squeeze(pred).asnumpy()  # [1, 1, H, W] -> [H, W]

    pred = cv2.resize(pred, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    pred_mask = np.where(pred > threshold, 1, 0)
    origin_img = np.array(origin_img, dtype=np.uint8)
    seg_img = origin_img * pred_mask[..., None]
    plt.imshow(seg_img)
    plt.show()
    cv2.imwrite("pred_result.png", cv2.cvtColor(
        seg_img.astype(np.uint8), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
