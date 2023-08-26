'''测试vit模型'''
# pylint:disable=E0401
import os
import numpy as np
import mindspore
import mindspore.dataset as ds
from PIL import Image
import matplotlib.pyplot as plt
from utils import GradCAM, show_cam_on_image, center_crop_img
from vit_model import vit_b_16_384


class ReshapeTransform:
    '''图像处理'''

    def __init__(self, model):
        input_size = model.patch_embed.image_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def main():
    '''主函数'''
    model = vit_b_16_384()
    weights_path = "./vit_base_patch16_224.ckpt"
    param_dict = mindspore.load_checkpoint(weights_path)
    # 将参数加载到网络中
    mindspore.load_param_into_net(model, param_dict)
    # Since the final classification is done on the class token computed in the last attention block,
    # the output will not be affected by the 14x14 channels in the last layer.
    # The gradient of the output with respect to them, will be 0!
    # We should chose any layer before the final attention block.
    target_layers = [model.backbone.transformer.norm1s[-1]]

    data_transform = ds.transforms.Compose([ds.vision.ToTensor(),
                                            ds.vision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # load image
    img_path = "both.png"
    assert os.path.exists(
        img_path), f"file: '{img_path}' dose not exist."
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)
    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = mindspore.ops.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  reshape_transform=ReshapeTransform(model))
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
