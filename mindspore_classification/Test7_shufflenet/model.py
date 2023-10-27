'''model'''
from typing import List, Callable

import mindspore
from mindspore import Tensor
import mindspore.nn as nn


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    '''channels'''
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = mindspore.ops.swapaxes(x, 1, 2)

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual(nn.Cell):
    '''InvertedResidual'''

    def __init__(self, input_c: int, output_c: int, stride: int):
        super().__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.SequentialCell(
                self.depthwise_conv(
                    input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1,
                          stride=1, padding=0, has_bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU()
            )
        else:
            self.branch1 = nn.SequentialCell()

        self.branch2 = nn.SequentialCell(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, has_bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(),
            self.depthwise_conv(branch_features, branch_features,
                                kernel_s=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features,
                      kernel_size=1, stride=1, padding=0, has_bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU()
        )

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        '''Conv'''
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, has_bias=bias, group=input_c)

    def construct(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = mindspore.ops.cat((x1, self.branch2(x2)), axis=1)
        else:
            out = mindspore.ops.cat((self.branch1(x), self.branch2(x)), axis=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Cell):
    '''ShuffleNetV2'''

    def __init__(self,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Cell] = InvertedResidual):
        super().__init__()

        if len(stages_repeats) != 3:
            raise ValueError(
                "expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError(
                "expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.SequentialCell(
            nn.Conv2d(input_channels, output_channels, kernel_size=3,
                      stride=2, padding=1, has_bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.SequentialCell
        self.stage3: nn.SequentialCell
        self.stage4: nn.SequentialCell

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                seq.append(inverted_residual(
                    output_channels, output_channels, 1))
            setattr(self, name, nn.SequentialCell(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.SequentialCell(
            nn.Conv2d(input_channels, output_channels, kernel_size=1,
                      stride=1, padding=0, has_bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

        self.fc = nn.Dense(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global pool
        x = self.fc(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def shufflenet_v2_x0_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 48, 96, 192, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x1_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 116, 232, 464, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x1_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 176, 352, 704, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x2_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 244, 488, 976, 2048],
                         num_classes=num_classes)

    return model
