"""
original code from facebook research:
https://github.com/facebookresearch/ConvNeXt
"""
import numpy as np
from mindspore import Parameter, Tensor, nn, ops
from mindspore import dtype as mstype
from mindspore.common import initializer as weight_init


class Identity(nn.Cell):
    """Identity"""

    def construct(self, x):
        '''Identity construct'''
        return x


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob, ndim=2):
        super().__init__()
        self.drop = nn.Dropout(keep_prob=1 - drop_prob)
        shape = (1,) + (1,) * (ndim + 1)
        self.ndim = ndim
        self.mask = Tensor(np.ones(shape), dtype=mstype.float32)

    def construct(self, x):
        '''droppath construct'''
        if not self.training:
            return x
        mask = ops.Tile()(self.mask, (x.shape[0],) + (1,) * (self.ndim + 1))
        out = self.drop(mask)
        out = out * x
        return out


class ConvNextLayerNorm(nn.LayerNorm):
    """ConvNextLayerNorm"""

    def __init__(self, normalized_shape, epsilon, norm_axis=-1):
        super().__init__(
            normalized_shape=normalized_shape, epsilon=epsilon)
        assert norm_axis in (-1,
                             1), "ConvNextLayerNorm's norm_axis must be 1 or -1."
        self.norm_axis = norm_axis

    def construct(self, input_x):
        '''ConvNextLayerNorm construct'''
        if self.norm_axis == -1:
            y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
        else:
            input_x = ops.Transpose()(input_x, (0, 2, 3, 1))
            y, _, _ = self.layer_norm(input_x, self.gamma, self.beta)
            y = ops.Transpose()(y, (0, 3, 1, 2))
        return y


class Block(nn.Cell):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Dense -> GELU -> Dense; Permute back
    We use (2) as we find it slightly faster in PyTorch
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim,
                 drop_path: float = 0.,
                 layer_scale_init_value: float = 1e-6):
        super().__init__()
        # MindSpore和torch都是用全连接层实现1*1卷积和逐点卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7,
                                group=dim, has_bias=True)  # depthwise conv
        self.norm = ConvNextLayerNorm((dim,), epsilon=1e-6)
        # pointwise/1x1 convs, implemented with Dense layers
        self.pwconv1 = nn.Dense(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Dense(4 * dim, dim)

        self.gamma = Parameter(Tensor(layer_scale_init_value * np.ones(dim), dtype=mstype.float32),
                               requires_grad=True) if layer_scale_init_value > 0 else None
        # 对于训练时随机丢弃网络节点的拓扑结构变化操作，MindSpore使用的DropPath算子，而torch使用的是随机深度算子
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else Identity()

    def construct(self, x):
        """Block construct"""
        downsample = x
        x = self.dwconv(x)
        x = ops.transpose(x, (0, 2, 3, 1))
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = ops.Transpose()(x, (0, 3, 1, 2))
        x = downsample + self.drop_path(x)
        return x


class ConvNeXt(nn.Cell):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()
        # stem and 3 intermediate downsampling conv layers
        # stem and 3 intermediate down_sampling conv layers
        self.downsample_layers = nn.CellList()
        stem = nn.SequentialCell(
            nn.Conv2d(in_chans, dims[0], kernel_size=4,
                      stride=4, has_bias=True),
            ConvNextLayerNorm((dims[0],), epsilon=1e-6, norm_axis=1)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.SequentialCell(
                ConvNextLayerNorm((dims[i],), epsilon=1e-6, norm_axis=1),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2,
                          stride=2, has_bias=True),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.CellList()
        dp_rates = list(x for x in np.linspace(0, drop_path_rate, sum(depths)))
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(Block(dim=dims[i], drop_path=dp_rates[cur + j],
                                    layer_scale_init_value=layer_scale_init_value))
            stage = nn.SequentialCell(blocks)
            self.stages.append(stage)
            cur += depths[i]

        self.norm = ConvNextLayerNorm(
            (dims[-1],), epsilon=1e-6)  # final norm layer
        self.head = nn.Dense(dims[-1], num_classes)  # classifier

        self.init_weights()
        self.head.weight.set_data(self.head.weight * head_init_scale)
        self.head.bias.set_data(self.head.bias * head_init_scale)

    def init_weights(self):
        """init_weights"""
        for _, cell in self.cells_and_names():
            if isinstance(cell, (nn.Dense, nn.Conv2d)):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))

    def forward_features(self, x):
        '''特征提取'''
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # global average pooling, (N, C, H, W) -> (N, C)
        return self.norm(x.mean([-2, -1]))

    def construct(self, x):
        '''ConvNeXt construct'''
        x = self.forward_features(x)
        x = self.head(x)
        return x


def convnext_tiny(num_classes: int):
    '''convnext_tiny'''
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def convnext_small(num_classes: int):
    '''convnext_small'''
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model


def convnext_base(num_classes: int):
    '''convnext_base'''
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes)
    return model


def convnext_large(num_classes: int):
    '''convnext_large'''
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes)
    return model


def convnext_xlarge(num_classes: int):
    '''convnext_xlarge'''
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes)
    return model
