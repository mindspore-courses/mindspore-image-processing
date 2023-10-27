'''预测'''
# pylint: disable=E0401
import os
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import mindspore

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs
import transforms


def create_model(num_classes, box_thresh=0.5):
    '''model'''
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def main():
    '''main'''
    mindspore.set_context(device_target="GPU")
    num_classes = 90  # 不包含背景
    box_thresh = 0.5
    weights_path = "./save_weights/model_25.ckpt"
    img_path = "./test.jpg"
    label_json_path = './coco91_indices.json'

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(
        weights_path), "{} file dose not exist.".format(weights_path)
    weights = mindspore.load_checkpoint(weights_path)
    mindspore.load_param_into_net(model, weights)
    model.set_train(False)

    # read class_indict
    assert os.path.exists(
        label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # load image
    assert os.path.exists(img_path), f"{img_path} does not exits."
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = mindspore.ops.unsqueeze(img, dim=0)

    # init
    img_height, img_width = img.shape[-2:]
    init_img = mindspore.ops.zeros(
        (1, 3, img_height, img_width))
    model(init_img)

    predictions = model(img)[0]

    predict_boxes = predictions["boxes"].asnumpy()
    predict_classes = predictions["labels"].asnumpy()
    predict_scores = predictions["scores"].asnumpy()
    predict_mask = predictions["masks"].asnumpy()
    # [batch, 1, h, w] -> [batch, h, w]
    predict_mask = np.squeeze(predict_mask, axis=1)

    if len(predict_boxes) == 0:
        print("没有检测到任何目标!")
        return

    plot_img = draw_objs(original_img,
                         boxes=predict_boxes,
                         classes=predict_classes,
                         scores=predict_scores,
                         masks=predict_mask,
                         category_index=category_index,
                         line_thickness=3,
                         font='arial.ttf',
                         font_size=20)
    plt.imshow(plot_img)
    plt.show()
    # 保存预测的图片结果
    plot_img.save("test_result.jpg")


if __name__ == '__main__':
    main()
