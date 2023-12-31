'''模型性能比较'''
# pylint: disable=E0401
import time
import numpy as np
import mindspore
import mindspore_hub as mshub
import onnxruntime
import matplotlib.pyplot as plt
from openvino.runtime import Core


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize the image to the given mean and standard deviation
    """
    image = image.astype(np.float32)
    image /= 255.0
    return image


def onnx_inference(onnx_path: str, image: np.ndarray, num_images: int = 20):
    '''onnx接口'''
    # load onnx model
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # compute onnx Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: image}

    start = time.perf_counter()
    for _ in range(num_images):
        ort_session.run(None, ort_inputs)
    end = time.perf_counter()
    time_onnx = end - start
    print(
        f"ONNX model in Inference Engine/CPU: {time_onnx / num_images:.3f} "
        f"seconds per image, FPS: {num_images / time_onnx:.2f}"
    )

    return num_images / time_onnx


def ir_inference(ir_path: str, image: np.ndarray, num_images: int = 20):
    '''openvion ir接口'''
    # Load the network in Inference Engine
    ie = Core()
    model_ir = ie.read_model(model=ir_path)
    compiled_model_ir = ie.compile_model(model=model_ir, device_name="CPU")

    # Get input and output layers
    input_layer_ir = next(iter(compiled_model_ir.inputs))
    _ = next(iter(compiled_model_ir.outputs))

    start = time.perf_counter()
    request_ir = compiled_model_ir.create_infer_request()
    for _ in range(num_images):
        request_ir.infer(inputs={input_layer_ir.any_name: image})
    end = time.perf_counter()
    time_ir = end - start
    print(
        f"IR model in Inference Engine/CPU: {time_ir / num_images:.3f} "
        f"seconds per image, FPS: {num_images / time_ir:.2f}"
    )

    return num_images / time_ir


def mindspore_inference(image: np.ndarray, num_images: int = 20):
    '''mindspore接口'''
    image = mindspore.Tensor(image, dtype=mindspore.float32)

    model = "mindspore/1.9/yolov5s_coco2017"
    model = mshub.load(model)
    model.set_train(False)

    start = time.perf_counter()
    for _ in range(num_images):
        model(image)
    end = time.perf_counter()
    time_torch = end - start

    print(
        f"mindspore model on CPU: {time_torch / num_images:.3f} seconds per image, "
        f"FPS: {num_images / time_torch:.2f}"
    )

    return num_images / time_torch


def plot_fps(pv: dict):
    '''绘制帧率（FPS）比较图'''
    x = list(pv.keys())
    y = list(pv.values())

    plt.bar(range(len(x)), y, align='center')
    plt.xticks(range(len(x)), x)
    for i, v in enumerate(y):
        plt.text(x=i, y=v+0.5, s=f"{v:.2f}", ha='center')
    plt.xlabel('model format')
    plt.ylabel('fps')
    plt.title('FPS comparison')
    plt.show()
    plt.savefig('fps_vs.jpg')


def main():
    '''主程序'''
    image_h = 640
    image_w = 640
    onnx_path = "yolov5s.onnx"
    ir_path = "ir_output/yolov5s.xml"

    image = np.random.randn(image_h, image_w, 3)
    normalized_image = normalize(image)

    # Convert the resized images to network input shape
    # [h, w, c] -> [c, h, w] -> [1, c, h, w]
    input_image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
    normalized_input_image = np.expand_dims(
        np.transpose(normalized_image, (2, 0, 1)), 0)

    onnx_fps = onnx_inference(
        onnx_path, normalized_input_image, num_images=100)
    ir_fps = ir_inference(ir_path, input_image, num_images=100)
    mindspore_fps = mindspore_inference(normalized_input_image, num_images=100)
    plot_fps({"mindspore": round(mindspore_fps, 2),
              "onnx": round(onnx_fps, 2),
              "ir": round(ir_fps, 2)})


if __name__ == '__main__':
    main()
