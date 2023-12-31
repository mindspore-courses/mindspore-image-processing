'''主函数'''
# pylint: disable = E0401
import os
import io
import json
import numpy as np
import mindspore
import mindspore.context as context
import cv2
from PIL import Image
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from model import MobileNetV2

app = Flask(__name__)
CORS(app)  # 解决跨域问题

weights_path = "./MobileNetV2(flower).ckpt"
class_json_path = "./class_indices.json"
assert os.path.exists(weights_path), "weights path does not exist..."
assert os.path.exists(class_json_path), "class json path does not exist..."

# select device
context.set_context(device_target="GPU")

# create model
model = MobileNetV2(num_classes=5)
# load model weights
param_not_load, _ = mindspore.load_param_into_net(model,
                                                  mindspore.load_checkpoint(weights_path))
print(param_not_load)
model.set_train(False)

# load class info
json_file = open(class_json_path, 'rb')
class_indict = json.load(json_file)


def transform_image(image_bytes):
    '''图像变换'''
    image = Image.open(io.BytesIO(image_bytes))
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
    return image.unsqueeze(0)


def get_prediction(image_bytes):
    '''返回结果'''
    try:
        tensor = transform_image(image_bytes=image_bytes)
        outputs = mindspore.ops.softmax(
            model.construct(tensor).squeeze(), axis=0)
        prediction = outputs.detach().cpu().numpy()
        template = "class:{:<15} probability:{:.3f}"
        index_pre = [(class_indict[str(index)], float(p))
                     for index, p in enumerate(prediction)]
        # sort probability
        index_pre.sort(key=lambda x: x[1], reverse=True)
        text = [template.format(k, v) for k, v in index_pre]
        return_info = {"result": text}
    except Exception as e:
        return_info = {"result": [str(e)]}
    return return_info


@app.route("/predict", methods=["POST"])
def predict():
    '''预测'''
    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes)
    return jsonify(info)


@app.route("/", methods=["GET", "POST"])
def root():
    '''页面'''
    return render_template("up.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
