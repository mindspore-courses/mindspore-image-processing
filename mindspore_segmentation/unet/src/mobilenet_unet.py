'''model'''
# pylint: disable = E0401
from collections import OrderedDict
from typing import Dict
from mindspore import nn, ops, Tensor
from mindcv.models import mobilenet_v3_large_100
from .unet import Up, OutConv


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


class MobileV3Unet(nn.Cell):
    '''MobileV3Unet'''

    def __init__(self, num_classes, pretrain_backbone: bool = False):
        super().__init__()
        backbone = mobilenet_v3_large_100(pretrained=pretrain_backbone)

        backbone = backbone.features

        stage_indices = [1, 3, 6, 12, 15]
        self.stage_out_channels = [
            backbone[i].out_channels for i in stage_indices]
        return_layers = dict(
            (str(j), f"stage{i}") for i, j in enumerate(stage_indices))
        self.backbone = IntermediateLayerGetter(
            backbone, return_layers=return_layers)

        c = self.stage_out_channels[4] + self.stage_out_channels[3]
        self.up1 = Up(c, self.stage_out_channels[3])
        c = self.stage_out_channels[3] + self.stage_out_channels[2]
        self.up2 = Up(c, self.stage_out_channels[2])
        c = self.stage_out_channels[2] + self.stage_out_channels[1]
        self.up3 = Up(c, self.stage_out_channels[1])
        c = self.stage_out_channels[1] + self.stage_out_channels[0]
        self.up4 = Up(c, self.stage_out_channels[0])
        self.conv = OutConv(
            self.stage_out_channels[0], num_classes=num_classes)

    def construct(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        backbone_out = self.backbone(x)
        x = self.up1(backbone_out['stage4'], backbone_out['stage3'])
        x = self.up2(x, backbone_out['stage2'])
        x = self.up3(x, backbone_out['stage1'])
        x = self.up4(x, backbone_out['stage0'])
        x = self.conv(x)
        x = ops.interpolate(x, size=input_shape,
                            mode="bilinear", align_corners=False)

        return {"out": x}
