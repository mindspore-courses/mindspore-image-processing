'''权重参数分析'''
# pylint:disable=W0611,E0401
import mindspore
from alexnet_model import AlexNet
from resnet_model import resnet34
import matplotlib.pyplot as plt
import numpy as np


# create model
model = AlexNet(num_classes=5)
# model = resnet34(num_classes=5)
# load model weights
model_weight_path = "./AlexNet.ckpt"  # "resNet34.ckpt"
param_not_load, _ = mindspore.load_param_into_net(model,
                                                  mindspore.load_checkpoint(model_weight_path))
print(param_not_load)
print(model)
model.set_train(False)

weights_keys = model.parameters_dict().keys()
for key in weights_keys:
    # remove num_batches_tracked para(in bn)
    if "num_batches_tracked" in key:
        continue
    # [kernel_number, kernel_channel, kernel_height, kernel_width]
    weight_t = model.parameters_dict()[key].numpy()

    # read a kernel information
    # k = weight_t[0, :, :, :]

    # calculate mean, std, min, max
    weight_mean = weight_t.mean()
    weight_std = weight_t.std(ddof=1)
    weight_min = weight_t.min()
    weight_max = weight_t.max()
    print(
        f"mean is {weight_mean}, std is {weight_std}, min is {weight_min}, max is {weight_max}")

    # plot hist image
    plt.close()
    weight_vec = np.reshape(weight_t, [-1])
    plt.hist(weight_vec, bins=50)
    plt.title(key)
    plt.show()
