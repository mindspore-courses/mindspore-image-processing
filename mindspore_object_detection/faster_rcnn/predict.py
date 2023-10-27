'''预测'''
# pylint: disable=E0401, W0611
import os
import json

import mindspore as ms
from PIL import Image
import matplotlib.pyplot as plt

from mindspore import dataset
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_objs


def create_model(num_classes):
    '''create model'''
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=ms.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def main():
    '''main'''
    ms.set_context(device_target="GPU")

    # create model
    model = create_model(num_classes=21)

    # load train weights
    weights_path = "./save_weights/model.pth"
    assert os.path.exists(
        weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = ms.load_checkpoint(weights_path)
    ms.load_param_into_net(model, weights_dict)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(
        label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load image
    original_img = Image.open("./test.jpg")

    # from pil image to tensor, do not normalize image
    data_transform = dataset.transforms.Compose([dataset.vision.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = ms.ops.unsqueeze(img, dim=0)

    model.set_train(False)  # 进入验证模式

    # init
    img_height, img_width = img.shape[-2:]
    init_img = ms.ops.zeros((1, 3, img_height, img_width))
    model(init_img)

    predictions = model(img)[0]

    predict_boxes = predictions["boxes"].asnumpy()
    predict_classes = predictions["labels"].asnumpy()
    predict_scores = predictions["scores"].asnumpy()

    if len(predict_boxes) == 0:
        print("没有检测到任何目标!")

    plot_img = draw_objs(original_img,
                         predict_boxes,
                         predict_classes,
                         predict_scores,
                         category_index=category_index,
                         box_thresh=0.5,
                         line_thickness=3,
                         font='arial.ttf',
                         font_size=20)
    plt.imshow(plot_img)
    plt.show()
    # 保存预测的图片结果
    plot_img.save("test_result.jpg")


if __name__ == '__main__':
    main()
