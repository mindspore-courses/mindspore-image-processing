'''测试'''
# pylint: disable=E0401, E0611
import os
import time
import mindspore
import mindspore.dataset as ds
from tqdm import tqdm
import onnx
import onnxruntime
import numpy as np
from openvino.inference_engine import IECore
from model import resnet34

mindspore.context.set_context(device_target="CPU")


def check_path_exist(path):
    '''检查路径'''
    assert os.path.exists(path), f"{path} does not exist..."


def to_numpy(tensor):
    '''转为numpy类型'''
    return tensor.asnumpy()


def mindspore_model_speed(data_loader, val_num):
    '''mindspore框架'''
    net = resnet34(num_classes=5)
    # load weights
    model_weight_path = "./resNet34.ckpt"
    check_path_exist(model_weight_path)

    param_not_load, _ = mindspore.load_param_into_net(
        net, mindspore.load_checkpoint(model_weight_path))
    print(param_not_load)
    net.set_train(False)
    test_data = mindspore.ops.rand((1, 3, 224, 224))
    net(test_data)

    forward_time = 0
    acc = 0.0  # accumulate accurate number / epoch

    for val_data in tqdm(data_loader, desc="Running pytorch model..."):
        val_images, val_labels = val_data
        t1 = time.time()
        # eval model only have last output layer
        outputs = net(val_images)
        t2 = time.time()
        forward_time += (t2 - t1)
        predict_y = mindspore.ops.max(outputs, axis=1)[1]
        acc += (predict_y == val_labels).sum().item()
    val_accurate = acc / val_num
    fps = round(val_num / forward_time, 1)
    print(f"mindspore info: fps: {fps}/s  accuracy: {val_accurate}")
    return fps, val_accurate, "Mindspore(not opt)"


def onnx_model_speed(data_loader, val_num):
    '''onnx格式模型'''
    # check onnx model
    onnx_path = "./resnet34.onnx"
    check_path_exist(onnx_path)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    forward_time = 0
    acc = 0.0  # accumulate accurate number / epoch
    for val_data in tqdm(data_loader, desc="Running onnx model..."):
        val_images, val_labels = val_data
        input_dict = {input_name: to_numpy(val_images)}
        t1 = time.time()
        outputs = ort_session.run(None, input_dict)
        t2 = time.time()
        forward_time += (t2 - t1)
        outputs = outputs[0]
        predict_y = np.argmax(outputs, axis=1)
        acc += (predict_y == to_numpy(val_labels)).sum()
    val_accurate = acc / val_num
    fps = round(val_num / forward_time, 1)
    print(f"onnx info:\nfps: {fps}/s  accuracy: {val_accurate}\n")
    return fps, val_accurate, "ONNX"


def openvino_model_speed(data_loader, val_num):
    '''openvino模型'''
    device = "CPU"
    model_xml_path = "./resnet34r.xml"
    model_bin_path = "./resnet34r.bin"
    check_path_exist(model_xml_path)
    check_path_exist(model_bin_path)

    # inference engine
    ie = IECore()

    # read IR
    net = ie.read_network(model=model_xml_path, weights=model_bin_path)
    # load model
    exec_net = ie.load_network(network=net, device_name=device)

    # check supported layers for device
    if device == "CPU":
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [
            l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) > 0:
            print("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                  "or --cpu_extension command line argument")
            raise ValueError("device {} not support layers:\n {}".format(device,
                                                                         ",".join(not_supported_layers)))

    # get input and output name
    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))

    # set batch size
    batch_size = 1
    net.batch_size = batch_size

    # read and pre-process input images
    # n, c, h, w = net.input_info[input_blob].input_data.shape
    forward_time = 0
    acc = 0.0  # accumulate accurate number / epoch
    for val_data in tqdm(data_loader, desc="Running onnx model..."):
        val_images, val_labels = val_data
        input_dict = {input_blob: to_numpy(val_images)}
        # start sync inference
        t1 = time.time()
        res = exec_net.infer(inputs=input_dict)
        t2 = time.time()
        forward_time += (t2 - t1)
        outputs = res[output_blob]
        predict_y = np.argmax(outputs, axis=1)
        acc += (predict_y == to_numpy(val_labels)).sum()
    val_accurate = acc / val_num
    fps = round(val_num / forward_time, 1)
    print("openvino info:\nfps: {}/s  accuracy: {}\n".format(fps,
                                                             val_accurate))


def main():
    '''主函数'''
    data_transform = ds.transforms.Compose([ds.vision.Resize(224),
                                            ds.vision.ToTensor(),
                                            ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = "/home/w180662/my_project/my_github"  # get data root path
    image_path = os.path.join(
        data_root, "data_set/flower_data/")  # flower data set path
    check_path_exist(image_path)

    batch_size = 1

    validate_dataset = ds.ImageFolderDataset(dataset_dir=image_path + "val")
    validate_dataset = validate_dataset.map(
        operations=data_transform, input_columns=["image"])
    val_num = len(validate_dataset)
    validate_loader = ds.GeneratorDataset(validate_dataset,
                                          shuffle=False,
                                          num_parallel_workers=4)
    validate_loader = validate_loader.batch(batch_size=batch_size)

    mindspore_model_speed(validate_loader, val_num)
    onnx_model_speed(validate_loader, val_num)
    openvino_model_speed(validate_loader, val_num)


if __name__ == '__main__':
    main()
