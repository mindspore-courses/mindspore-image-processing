'''模型预测'''
# pylint: disable = E0401
import sys
import json
from model import resnet34
from PIL import Image
import cv2
import mindspore
import mindspore.context as context
import mindspore.ops as ops
import matplotlib.pyplot as plt

context.set_context(device_target="GPU")

# load image
img = Image.open("../tulip.jpg")
plt.imshow(img)
# [N, C, H, W]
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img = cv2.resize(img, [224, 224])
w, h = img.shape[1], img.shape[0]
img = img[(h*0.25):(h*0.75), (w*0.25):(w*0.75)]
img = cv2.resize(img, [224, 224])
img = (img[:, :] - mean) / std
img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
img = mindspore.Tensor(img)
# expand batch dimension
img = ops.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('./class_indices.json', 'r', encoding='utf-8')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    sys.exit(-1)

# create model
model = resnet34(num_classes=5)
# load model weights
model_weight_path = "./resNet34.ckpt"
param_not_load, _ = mindspore.load_param_into_net(
    model, mindspore.load_checkpoint(model_weight_path))
print(param_not_load)
model.set_train(False)
# predict class
output = ops.squeeze(model(img))
predict = ops.softmax(output, axis=0)
predict_cla = ops.argmax(predict).numpy()
print(class_indict[str(predict_cla)], predict[predict_cla].numpy())
plt.show()
