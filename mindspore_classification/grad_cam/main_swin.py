'''测试swin模型'''
# pylint:disable=E0401
import os
import math
import numpy as np
import mindspore
import mindspore.dataset as ds
from PIL import Image
import matplotlib.pyplot as plt

from utils import GradCAM, show_cam_on_image, center_crop_img
from swin_model import swinv2_base_window7


class ResizeTransform:
    '''图像处理'''

    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        '''尺寸变化'''
        s = math.ceil(s / 4)  # PatchEmbed
        s = math.ceil(s / 2)  # PatchMerging1
        s = math.ceil(s / 2)  # PatchMerging2
        s = math.ceil(s / 2)  # PatchMerging3
        return s

    def __call__(self, x):
        result = x.reshape(x.size(0),
                           self.height,
                           self.width,
                           x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)

        return result


def main():
    '''主函数'''
    # 注意输入的图片必须是32的整数倍
    # 否则由于padding的原因会出现注意力飘逸的问题
    img_size = 224
    assert img_size % 32 == 0

    model = swinv2_base_window7()
    weights_path = "./swinv2_base_window7.ckpt"
    param_dict = mindspore.load_checkpoint(weights_path)
    # 将参数加载到网络中
    mindspore.load_param_into_net(model, param_dict)

    target_layers = [model.norm]

    data_transform = ds.transforms.Compose([ds.vision.ToTensor(),
                                            ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "both.png"
    assert os.path.exists(
        img_path), f"file: '{img_path}' dose not exist."
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, img_size)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = mindspore.ops.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=ResizeTransform(im_h=img_size, im_w=img_size))
    target_category = 281  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
