'''VGG'''
# coding:utf8
# pylint: disable=E0401
import mindspore.nn as nn
from mindspore.common import initializer as weight_init


class VGG(nn.Cell):
    '''VGG'''

    def __init__(self, features, num_classes=1000, init_weights=False):
        super().__init__()
        self.features = features
        self.classifier = nn.SequentialCell(
            nn.Dense(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Dense(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def construct(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = x.view(x.size(0), 512 * 7 * 7)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        '''初始化'''
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.01),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.01),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.bias.shape,
                                                           cell.bias.dtype))


def make_features(cfg: list):
    '''特征层'''
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               padding=1, has_bias=True)
            layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    '''返回VGG模型'''
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(
        model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model
