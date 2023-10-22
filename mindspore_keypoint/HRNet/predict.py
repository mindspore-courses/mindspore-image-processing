'''预测'''
import os
import json

import mindspore
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import HighResolutionNet
from draw_utils import draw_keypoints
import transforms


def predict_single_person():
    '''main'''
    mindspore.set_context(device_target="GPU")

    flip_test = True
    resize_hw = (256, 192)
    img_path = "./person.png"
    weights_path = "./pose_hrnet_w32_256x192.ckpt"
    keypoint_json_path = "person_keypoints.json"
    assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(
        weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(
        keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    # read json file
    with open(keypoint_json_path, "r", encoding='utf-8') as f:
        person_info = json.load(f)

    # read single-person image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor, target = data_transform(
        img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    img_tensor = mindspore.ops.unsqueeze(img_tensor, dim=0)

    # create model
    # HRNet-W32: base_channel=32
    # HRNet-W48: base_channel=48
    model = HighResolutionNet(base_channel=32)
    weights = mindspore.load_checkpoint(weights_path)
    mindspore.load_param_into_net(model, weights)
    model.set_train(False)

    # with torch.inference_mode():
    outputs = model(img_tensor)

    if flip_test:
        flip_tensor = transforms.flip_images(img_tensor)
        flip_outputs = mindspore.ops.squeeze(
            transforms.flip_back(
                model(flip_tensor), person_info["flip_pairs"]),
        )
        # feature is not aligned, shift flipped heatmap for higher accuracy
        # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
        flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
        outputs = (outputs + flip_outputs) * 0.5

        keypoints, scores = transforms.get_final_preds(
            outputs, [target["reverse_trans"]], True)
        keypoints = np.squeeze(keypoints)
        scores = np.squeeze(scores)

        plot_img = draw_keypoints(img, keypoints, scores, thresh=0.2, r=3)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save("test_result.jpg")


if __name__ == '__main__':
    predict_single_person()
