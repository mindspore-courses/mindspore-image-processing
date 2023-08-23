'''CNN模型'''
import os
import numpy as np
import mindspore
from PIL import Image
import matplotlib.pyplot as plt
import mindcv.models as models
from utils import GradCAM, show_cam_on_image, center_crop_img


def main():
    '''主函数'''
    model = models.mobilenet_v3_large_075(pretrained=True)
    target_layers = [model.features[-1]]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = mindspore.dataset.transforms.Compose([mindspore.dataset.vision.ToTensor(),
                                                           mindspore.dataset.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "both.png"
    assert os.path.exists(
        img_path), f"file: '{img_path}' dose not exist."
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = mindspore.ops.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers)
    target_category = 281  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
