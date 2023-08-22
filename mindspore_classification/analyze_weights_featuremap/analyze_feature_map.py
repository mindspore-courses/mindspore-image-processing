'''特征图分析'''
# pylint:disable=W0611,E0401
import mindspore
import mindspore.dataset as ds
from alexnet_model import AlexNet
from resnet_model import resnet34
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2


def transform_image(image):
    '''图像变换'''
    if image.mode != "RGB":
        raise ValueError("input file does not RGB image...")
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = cv2.resize(image, [255, 255])
    image = (image[:, :] - mean) / std
    image = cv2.resize(image, [224, 224])
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = mindspore.Tensor(image)
    # expand batch dimension
    return image.unsqueeze(0)


# create model
model = AlexNet(num_classes=5)
# model = resnet34(num_classes=5)
# load model weights
model_weight_path = "./AlexNet.ckpt"  # "./resNet34.ckpt"
param_not_load, _ = mindspore.load_param_into_net(model,
                                                  mindspore.load_checkpoint(model_weight_path))
print(param_not_load)
print(model)
model.set_train(False)

# load image
img = Image.open("../tulip.jpg")
# [N, C, H, W]
img = transform_image(img)

# forward
out_put = model(img)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3, 4, i+1)
        # [H, W, C]
        plt.imshow(im[:, :, i], cmap='gray')
    plt.show()
