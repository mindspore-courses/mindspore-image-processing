'''主程序'''
from PIL import Image
import mindspore
import mindspore.context as context
import onnx
import onnxruntime
import cv2
import numpy as np
from model import resnet34

context.set_context(device_target="GPU")


def to_numpy(tensor):
    '''转为numpy类型'''
    return tensor.asnumpy()


def main(save_path=None):
    '''主函数'''
    assert isinstance(save_path, str), "lack of save_path parameter..."
    # create model
    model = resnet34(num_classes=5)
    # load model weights
    model_weight_path = "./resNet34.ckpt"
    param_not_load, _ = mindspore.load_param_into_net(model,
                                                      mindspore.load_checkpoint(model_weight_path))
    print(param_not_load)
    model.set_train(False)
    # input to the model
    # [batch, channel, height, width]
    x = mindspore.ops.rand((1, 3, 224, 224))
    torch_out = model(x)

    # export the model
    mindspore.export(net=model,                       # model being run
                     # model input (or a tuple for multiple inputs)
                     inputs=x,
                     # where to save the model (can be a file or file-like object)
                     file_name=save_path,
                     file_format='ONNX')

    # check onnx model
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(save_path)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and Pytorch results
    # assert_allclose: Raises an AssertionError if two objects are not equal up to desired tolerance.
    np.testing.assert_allclose(
        to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

    # load test image
    img = Image.open("../tulip.jpg")
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # pre-process
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = cv2.resize(img, [224, 224])
    img = (img[:, :] - mean) / std
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = mindspore.Tensor(img)

    img = img.unsqueeze(0)

    # feed image into onnx model
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)
    prediction = ort_outs[0]

    # np softmax process
    # 为了稳定地计算softmax概率， 一般会减掉最大元素
    prediction -= np.max(prediction, keepdims=True)
    prediction = np.exp(prediction) / np.sum(np.exp(prediction), keepdims=True)
    print(prediction)


if __name__ == '__main__':
    onnx_file_name = "resnet34.onnx"
    main(save_path=onnx_file_name)
