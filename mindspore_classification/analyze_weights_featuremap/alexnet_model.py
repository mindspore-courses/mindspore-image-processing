'''alexnet'''
import numpy as np
import mindspore as ms
import mindspore.nn as nn


class AlexNet(nn.Cell):
    """
    code from torchvision/models/alexnet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    """

    def __init__(self, num_classes=1000, init_weights=False):
        super().__init__()
        self.features = nn.SequentialCell(
            # input[3, 224, 224]  output[48, 55, 55]
            nn.Conv2d(3, 48, kernel_size=11, stride=4,
                      padding=2, has_bias=True),
            nn.ReLU(),
            # output[48, 27, 27]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[128, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2, has_bias=True),
            nn.ReLU(),
            # output[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[192, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1, has_bias=True),
            nn.ReLU(),
            # output[192, 13, 13]
            nn.Conv2d(192, 192, kernel_size=3, padding=1, has_bias=True),
            nn.ReLU(),
            # output[128, 13, 13]
            nn.Conv2d(192, 128, kernel_size=3, padding=1, has_bias=True),
            nn.ReLU(),
            # output[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.SequentialCell(
            nn.Dropout(),
            nn.Dense(128 * 6 * 6, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(2048, 2048),
            nn.ReLU(),
            nn.Dense(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def construct(self, x):
        """input: x"""
        outputs = []
        for name, module in self.features.cells_and_names():
            x = module(x)
            if name in ["0", "3", "6"]:
                outputs.append(x)

        return outputs

    def _initialize_weights(self):
        for cell in self.cells():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeNormal(
                        negative_slope=0, mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        ms.Tensor(np.zeros(cell.bias.data.shape, dtype="float32")))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(ms.Tensor(np.random.normal(
                    0, 0.01, cell.weight.data.shape).astype("float32")))
                if cell.bias is not None:
                    cell.bias.set_data(
                        ms.Tensor(np.zeros(cell.bias.data.shape, dtype="float32")))
