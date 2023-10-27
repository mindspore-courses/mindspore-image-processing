'''模型'''
# pylint:disable=E0401, E0602, W0401
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import mindspore.nn as nn
import mindspore
from mindspore import Tensor
import mindspore.common.initializer as init

from utils import *


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
        mindspore.ops.rand(shape, dtype=x.dtype)
    random_tensor = mindspore.ops.floor(random_tensor)  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Cell):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def construct(self, x):
        '''DropPath construct'''
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Cell):
    '''子模块'''

    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Cell]] = None,
                 activation_layer: Optional[Callable[..., nn.Cell]] = None):
        super().__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              group=groups,
                              has_bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def construct(self, x):
        '''ConvBNAct construct'''
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result

    def complexity(self, cx):
        '''ConvBNAct complexity'''
        cx = conv2d_cx(cx,
                       in_c=self.conv.in_channels,
                       out_c=self.conv.out_channels,
                       k=self.conv.kernel_size[0],  # tuple type
                       stride=self.conv.stride[0],  # tuple type
                       groups=self.conv.group,
                       bias=False,
                       trainable=self.conv.weight.requires_grad)
        cx = norm2d_cx(cx, self.conv.out_channels,
                       trainable=self.bn.weight.requires_grad)

        return cx


class SqueezeExcite(nn.Cell):
    '''子模块'''

    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.25):
        super().__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1, has_bias=True)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1, has_bias=True)
        self.act2 = nn.Sigmoid()

    def construct(self, x: Tensor) -> Tensor:
        '''SqueezeExcite construct'''
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x

    def complexity(self, cx):
        '''SqueezeExcite complexity'''
        h, w = cx["h"], cx["w"]
        cx = gap2d_cx(cx)
        cx = conv2d_cx(cx,
                       in_c=self.conv_reduce.in_channels,
                       out_c=self.conv_reduce.out_channels,
                       k=1,
                       bias=True,
                       trainable=self.conv_reduce.weight.requires_grad)
        cx = conv2d_cx(cx,
                       in_c=self.conv_expand.in_channels,
                       out_c=self.conv_expand.out_channels,
                       k=1,
                       bias=True,
                       trainable=self.conv_expand.weight.requires_grad)
        cx["h"], cx["w"] = h, w

        return cx


class MBConv(nn.Cell):
    '''子模块'''

    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Cell]):
        super().__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在
        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = ConvBNAct(input_c,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)

        # Depth-wise convolution
        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        self.se = SqueezeExcite(input_c, expanded_c,
                                se_ratio) if se_ratio > 0 else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_c,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity)  # 注意这里没有激活函数，所有传入Identity

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def construct(self, x: Tensor) -> Tensor:
        '''MBConv construct'''
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result

    def complexity(self, cx):
        '''MBConv complexity'''
        cx = self.expand_conv.complexity(cx)
        cx = self.dwconv.complexity(cx)
        cx = self.se.complexity(cx)
        cx = self.project_conv.complexity(cx)

        return cx


class FusedMBConv(nn.Cell):
    '''子模块'''

    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Cell]):
        super().__init__()

        assert stride in [1, 2]
        assert se_ratio == 0

        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 只有当expand ratio不等于1时才有expand conv
        if self.has_expansion:
            # Expansion convolution
            self.expand_conv = ConvBNAct(input_c,
                                         expanded_c,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.project_conv = ConvBNAct(expanded_c,
                                          out_c,
                                          kernel_size=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity)  # 注意没有激活函数
        else:
            # 当只有project_conv时的情况
            self.project_conv = ConvBNAct(input_c,
                                          out_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer)  # 注意有激活函数

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def construct(self, x: Tensor) -> Tensor:
        '''FusedMBConv construct'''
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)

            result += x

        return result

    def complexity(self, cx):
        '''FusedMBConv complexity'''
        if self.has_expansion:
            cx = self.expand_conv.complexity(cx)
            cx = self.project_conv.complexity(cx)
        else:
            cx = self.project_conv.complexity(cx)

        return cx


class EfficientNetV2(nn.Cell):
    '''模型'''

    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        super().__init__()

        for cnf in model_cnf:
            assert len(cnf) == 8
        self.model_cnf = model_cnf
        self.num_classes = num_classes
        self.num_features = num_features

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        stem_filter_num = model_cnf[0][4]

        self.stem = ConvBNAct(3,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)  # 激活函数默认是SiLU

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(op(kernel_size=cnf[1],
                                 input_c=cnf[4] if i == 0 else cnf[5],
                                 out_c=cnf[5],
                                 expand_ratio=cnf[3],
                                 stride=cnf[2] if i == 0 else 1,
                                 se_ratio=cnf[-1],
                                 drop_rate=drop_connect_rate * block_id / total_blocks,
                                 norm_layer=norm_layer))
                block_id += 1
        self.blocks = nn.SequentialCell(*blocks)

        head_input_c = model_cnf[-1][-3]
        head = OrderedDict()

        head.update({"project_conv": ConvBNAct(head_input_c,
                                               num_features,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})  # 激活函数默认是SiLU

        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate)})
        head.update({"classifier": nn.Dense(num_features, num_classes)})

        self.head = nn.SequentialCell(head)

        # initial weights
        for _,cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(
                    init.HeUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        "zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.weight.set_data(init.initializer(
                    "ones", cell.weight.shape, cell.weight.dtype))
                cell.bias.set_data(init.initializer(
                    "zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.TruncatedNormal(
                        sigma=0.01), cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        "zeros", cell.bias.shape, cell.bias.dtype))

    def construct(self, x: Tensor) -> Tensor:
        '''EfficientNetV2 construct'''
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x

    def complexity(self, h, w, c):
        '''EfficientNetV2 complexity'''
        cx = {"h": h, "w": w, "c": c, "flops": 0,
              "params": 0, "acts": 0, "freeze": 0}
        cx = self.stem.complexity(cx)

        for module in self.blocks.children():
            if hasattr(module, "complexity"):
                cx = module.complexity(cx)
            else:
                print(module)

        for module in self.head.children():
            if hasattr(module, "complexity"):
                cx = module.complexity(cx)
            elif isinstance(module, nn.Dense):
                in_units = module.in_features
                out_units = module.out_features
                cx = gap2d_cx(cx)
                cx = linear_cx(cx, in_units, out_units, bias=True,
                               trainable=module.weight.requires_grad)
        # print(cx)
        return cx


def efficientnetv2_s(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2)
    return model


def efficientnetv2_m(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3)
    return model


def efficientnetv2_l(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4)
    return model
