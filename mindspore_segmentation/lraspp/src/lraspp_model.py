'''model'''
from collections import OrderedDict

from typing import Dict

import mindspore as ms
from mindspore import nn, Tensor, ops
from .mobilenet_backbone import mobilenet_v3_large


class IntermediateLayerGetter(nn.CellDict):
    """
    Cell wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Cell
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Cell): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Cell, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.cells_and_names()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.cells_and_names():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def construct(self, x: Tensor) -> Dict[str, Tensor]:
        '''construct'''
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class LRASPP(nn.Cell):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.

    Args:
        backbone (nn.Cell): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int): number of output classes of the model (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """
    __constants__ = ['aux_classifier']

    def __init__(self,
                 backbone: nn.Cell,
                 low_channels: int,
                 high_channels: int,
                 num_classes: int,
                 inter_channels: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = LRASPPHead(
            low_channels, high_channels, num_classes, inter_channels)

    def construct(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        out = self.classifier(features)
        out = ops.interpolate(out, size=input_shape,
                              mode="bilinear", align_corners=False)

        result = OrderedDict()
        result["out"] = out

        return result


class LRASPPHead(nn.Cell):
    '''LRASPPHead'''

    def __init__(self,
                 low_channels: int,
                 high_channels: int,
                 num_classes: int,
                 inter_channels: int) -> None:
        super().__init__()
        self.cbr = nn.SequentialCell(
            nn.Conv2d(high_channels, inter_channels, 1, has_bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.scale = nn.SequentialCell(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, has_bias=False),
            nn.Sigmoid()
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def construct(self, inputs: Dict[str, Tensor]) -> Tensor:
        low = inputs["low"]
        high = inputs["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = ops.interpolate(
            x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)


def lraspp_mobilenetv3_large(num_classes=21, pretrain_backbone=False):
    '''model'''
    backbone = mobilenet_v3_large(dilated=True)

    if pretrain_backbone:
        # 载入mobilenetv3 large backbone预训练权重
        ms.load_param_into_net(
            backbone, ms.load_checkpoint('mobilenet_v3_large.ckpt'))

    backbone = backbone.features

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(
        b, "is_strided", False)] + [len(backbone) - 1]
    low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels

    return_layers = {str(low_pos): "low", str(high_pos): "high"}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = LRASPP(backbone, low_channels, high_channels, num_classes)
    return model
