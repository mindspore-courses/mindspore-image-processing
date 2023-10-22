'''model'''
import math

import mindspore.nn as nn
import mindspore
import mindspore.common.initializer as init


class VGG(nn.Cell):
    '''VGG'''

    def __init__(self, features, class_num=1000, init_weights=False, weights_path=None):
        super().__init__()
        self.features = features
        self.classifier = nn.SequentialCell(
            nn.Dense(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dense(4096, class_num)
        )
        if init_weights and weights_path is None:
            self._initialize_weights()

        if weights_path is not None:
            mindspore.load_param_into_net(
                self, mindspore.load_checkpoint(weights_path))

    def construct(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = mindspore.ops.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.HeNormal(math.sqrt(5), mode="fan_out", nonlinearity="relu"),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        "zeros", cell.bias.shape, cell.bias.dtype))


def make_features(cfg: list):
    '''features'''
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", weights_path=None):
    '''vgg'''
    assert model_name in cfgs, f"Warning: model number {model_name} not in cfgs dict!"
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), weights_path=weights_path)
    return model
