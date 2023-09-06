import os
import mindspore
import mindspore.nn as nn
from model import resnet34


def main():
    '''主函数'''
    mindspore.set_context(device_target="GPU")

    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet34-pre.pth"
    assert os.path.exists(
        model_weight_path), "file {} does not exist.".format(model_weight_path)

    # option1
    net = resnet34()
    param_dict = mindspore.load_checkpoint(model_weight_path)
    param_not_load, _ = mindspore.load_param_into_net(net, param_dict)
    print(param_not_load)
    net.set_train(False)
    # change fc layer structure
    in_channel = net.fc.in_channels
    net.fc = nn.Dense(in_channel, 5)

    # option2
    # net = resnet34(num_classes=5)
    # pre_weights = torch.load(model_weight_path, map_location=device)
    # del_key = []
    # for key, _ in pre_weights.items():
    #     if "fc" in key:
    #         del_key.append(key)
    #
    # for key in del_key:
    #     del pre_weights[key]
    #
    # missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    # print("[missing_keys]:", *missing_keys, sep="\n")
    # print("[unexpected_keys]:", *unexpected_keys, sep="\n")


if __name__ == '__main__':
    main()
