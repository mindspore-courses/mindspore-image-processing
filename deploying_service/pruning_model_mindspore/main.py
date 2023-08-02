import os
import time
import mindspore
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops

from tqdm import tqdm
from model import resnet34
from mindspore_gs import PrunerKfCompressAlgo, PrunerFtCompressAlgo

mindspore.context.set_context(device_target="GPU")

data_transform = ds.transforms.Compose([ds.vision.Resize(256),
                                        ds.vision.CenterCrop(224),
                                        ds.vision.ToTensor(),
                                        ds.vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data_root = os.path.abspath(os.path.join(
    os.getcwd(), "../.."))  # get data root path
image_path = data_root + "/data_set/flower_data/"  # flower data set path
batch_size = 16


def validate_model(model: nn.Cell):
    '''验证模型'''
    validate_dataset = ds.ImageFolderDataset(dataset_dir=image_path + "val",
                                             transform=data_transform)
    val_num = len(validate_dataset)
    validate_loader = ds.GeneratorDataset(validate_dataset,
                                          shuffle=False,
                                          num_parallel_workers=2)
    validate_loader = validate_loader.batch(batch_size=batch_size)

    model.set_train(False)
    acc = 0.0  # accumulate accurate number / epoch
    t1 = time.time()
    for val_data in tqdm(validate_loader, desc="validate model accuracy."):
        val_images, val_labels = val_data
        # eval model only have last output layer
        outputs = model(val_images)
        predict_y = ops.softmax(outputs, axis=0)[1]
        acc += ops.sum(ops.equal(predict_y, val_labels)).item()
    val_accurate = acc / val_num
    print('test_accuracy: {:3f}, time:{:3f}' %
          (val_accurate, time.time() - t1))

    return val_accurate


def count_sparsity(model: nn.Cell, p=True):
    sum_zeros_num = 0
    sum_weights_num = 0
    for name, module in model.cells_and_names():
        if isinstance(module, nn.Conv2d):
            zeros_elements = ops.sum(ops.equal(module.weight, 0)).item()
            weights_elements = module.weight.numel()

            sum_zeros_num += zeros_elements
            sum_weights_num += weights_elements
            if p is True:
                print(
                    f"Sparsity in {name}.weights {(100 * sum_zeros_num / sum_weights_num):.2f}%")
    print(f"Global sparsity: {(100 * sum_zeros_num / sum_weights_num):.2f}%")


def main():
    weights_path = "./resNet34.ckpt"
    model = resnet34(num_classes=5)
    param_not_load, _ = mindspore.load_param_into_net(
        model, mindspore.load_checkpoint(weights_path))
    print(param_not_load)

    # 对卷积核进行剪枝处理
    algo_kf = PrunerKfCompressAlgo({})
    model = algo_kf.apply(model)  # Get konckoff stage network
    # Define FineTune Algorithm
    ft_pruning = PrunerFtCompressAlgo({'prune_rate': 0.5})
    # Apply FineTune-algorithm to origin network
    model = ft_pruning.apply(model)

    # 统计剪枝比例
    count_sparsity(model, p=False)

    # 验证剪枝后的模型
    validate_model(model)


if __name__ == '__main__':
    main()
