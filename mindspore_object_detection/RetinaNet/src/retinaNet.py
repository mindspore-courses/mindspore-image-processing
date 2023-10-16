import math
import numpy as np
from functools import reduce

from mindspore import nn, ops, Tensor
from mindspore import context
import mindspore as ms
from mindspore.ops import functional, operations, composite
from mindspore.communication.management import get_group_size
from mindspore.parallel._auto_parallel_context import auto_parallel_context


def init_kaiming_uniform(arr_shape, a=0, nonlinearity='leaky_relu', has_bias=False):
    """
    Kaiming initialize, generate a tensor with input shape, according to He initialization, using a uniform
    distribution.
    """

    def _calculate_in_and_out(arr_shape):
        """Calculate input and output dimension of layer."""
        dim = len(arr_shape)

        n_in = arr_shape[1]
        n_out = arr_shape[0]

        if dim > 2:
            counter = reduce(lambda x, y: x * y, arr_shape[2:])
            n_in *= counter
            n_out *= counter
        return n_in, n_out

    def calculate_gain(nonlinearity, a=None):
        """Calculate gain of Kaiming initialization."""
        linear_fans = ['linear', 'conv1d', 'conv2d', 'conv3d',
                       'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
        if nonlinearity in linear_fans or nonlinearity == 'sigmoid':
            return 1
        if nonlinearity == 'tanh':
            return 5.0 / 3
        if nonlinearity == 'relu':
            return math.sqrt(2.0)
        if nonlinearity == 'leaky_relu':
            if a is None:
                negative_slope = 0.01
            elif not isinstance(a, bool) and isinstance(a, int) or isinstance(a, float):
                negative_slope = a
            return math.sqrt(2.0 / (1 + negative_slope ** 2))

        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

    fan_in, _ = _calculate_in_and_out(arr_shape)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    weight = np.random.uniform(-bound, bound, arr_shape).astype(np.float32)

    bias = None
    if has_bias:
        bound_bias = 1 / math.sqrt(fan_in)
        bias = np.random.uniform(-bound_bias, bound_bias,
                                 arr_shape[0:1]).astype(np.float32)
        bias = Tensor(bias)

    return Tensor(weight), bias


def conv(kernel_size, in_channel, out_channel, stride):
    """
    Convolution with particular kernel size.
    """
    weight_shape = (out_channel, in_channel, kernel_size, kernel_size)
    weight = _weight_variable(weight_shape)
    pad = (kernel_size - 1) // 2
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=kernel_size, stride=stride, padding=pad, pad_mode='pad', weight_init=weight)


def _fc(in_channel, out_channel):
    """
    Construct a fully connected layer with in_channel as input channel size, out_channel as output channel size.
    """
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class ResidualBlock(nn.Cell):
    """
    Residual block of resnet.
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock, self).__init__()

        channel = out_channel // self.expansion
        self.conv1 = conv(1, in_channel, channel, stride=1)
        self.bn1 = nn.BatchNorm2d(channel)

        self.conv2 = conv(3, channel, channel, stride=stride)
        self.bn2 = nn.BatchNorm2d(channel)

        self.conv3 = conv(1, channel, out_channel, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([conv(1, in_channel, out_channel, stride),
                                                        nn.BatchNorm2d(out_channel)])
        self.add = ops.Add()

    def construct(self, x):
        """Forward pass."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet structure.
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides):
        super(ResNet, self).__init__()

        self.conv1 = conv(7, 3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = ops.ReLU()

        self.zeros1 = ops.Zeros()
        self.zeros2 = ops.Zeros()
        self.concat1 = ops.Concat(axis=2)
        self.concat2 = ops.Concat(axis=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")

        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        """
        Construct a ResNet stage layer.
        """
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        zeros1 = self.zeros1(
            (x.shape[0], x.shape[1], 1, x.shape[3]), ms.float32)
        x = self.concat1((zeros1, x))
        zeros2 = self.zeros2(
            (x.shape[0], x.shape[1], x.shape[2], 1), ms.float32)
        x = self.concat2((zeros2, x))

        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c3, c4, c5


def resnet50():
    """Construct ResNet50 with 3 output stages."""
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2])


def _weight_variable(shape, factor=0.01):
    """Use standard normal distribution to initialize tensor."""
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


class ClassHead(nn.Cell):
    """
    RetinaNet ClassHead, judge whether anchor contains face.
    """

    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors

        weight_shape = (self.num_anchors * 2, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = init_kaiming_uniform(
            weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0,
                                 has_bias=True, weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """Forward pass."""
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (ops.Shape()(out)[0], -1, 2))


class BboxHead(nn.Cell):
    """
    RetinaNet BoxHead, predict height,width and center position of boxes.
    """

    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()

        weight_shape = (num_anchors * 4, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = init_kaiming_uniform(
            weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0, has_bias=True,
                                 weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """Forward pass."""
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (ops.Shape()(out)[0], -1, 4))


class LandmarkHead(nn.Cell):
    """
    RetinaNet BoxHead, predict position of landmarks.
    """

    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()

        weight_shape = (num_anchors * 10, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = init_kaiming_uniform(
            weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0, has_bias=True,
                                 weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """Forward pass."""
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (ops.Shape()(out)[0], -1, 10))


class ConvBNReLU(nn.SequentialCell):
    """Convolution,batch normalization and leaky ReLU with Kaiming uniform initialize."""

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, norm_layer):
        weight_shape = (out_planes, in_planes, kernel_size, kernel_size)
        kaiming_weight, _ = init_kaiming_uniform(weight_shape, a=math.sqrt(5))
        activation = nn.ReLU()
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', padding=padding, group=groups,
                      has_bias=False, weight_init=kaiming_weight),
            norm_layer(out_planes),
            activation
        )


class ConvBN(nn.SequentialCell):
    """Convolution and batch normalization with Kaiming uniform initialize."""

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, norm_layer):
        weight_shape = (out_planes, in_planes, kernel_size, kernel_size)
        kaiming_weight, _ = init_kaiming_uniform(weight_shape, a=math.sqrt(5))

        super(ConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', padding=padding, group=groups,
                      has_bias=False, weight_init=kaiming_weight),
            norm_layer(out_planes),
        )


class SSH(nn.Cell):
    """
    SSH feature pyramid structure.
    """

    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()

        norm_layer = nn.BatchNorm2d
        self.conv3x3 = ConvBN(in_channel, out_channel // 2, kernel_size=3, stride=1, padding=1, groups=1,
                              norm_layer=norm_layer)

        self.conv5x5_1 = ConvBNReLU(in_channel, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                    norm_layer=norm_layer)
        self.conv5x5_2 = ConvBN(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                norm_layer=norm_layer)

        self.conv7x7_2 = ConvBNReLU(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                    norm_layer=norm_layer)
        self.conv7x7_3 = ConvBN(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                norm_layer=norm_layer)

        self.cat = ops.Concat(axis=1)
        self.relu = nn.ReLU()

    def construct(self, x):
        """Forward pass."""
        conv3x3 = self.conv3x3(x)

        conv5x5_1 = self.conv5x5_1(x)
        conv5x5 = self.conv5x5_2(conv5x5_1)

        conv7x7_2 = self.conv7x7_2(conv5x5_1)
        conv7x7 = self.conv7x7_3(conv7x7_2)

        out = self.cat((conv3x3, conv5x5, conv7x7))
        out = self.relu(out)

        return out


class FPN(nn.Cell):
    """
    FPN feature pyramid structure.
    """

    def __init__(self, in_channel=None, out_channel=None):
        super(FPN, self).__init__()
        norm_layer = nn.BatchNorm2d
        if in_channel is None or out_channel is None:
            self.output1 = ConvBNReLU(512, 256, kernel_size=1, stride=1, padding=0, groups=1,
                                      norm_layer=norm_layer)
            self.output2 = ConvBNReLU(1024, 256, kernel_size=1, stride=1, padding=0, groups=1,
                                      norm_layer=norm_layer)
            self.output3 = ConvBNReLU(2048, 256, kernel_size=1, stride=1, padding=0, groups=1,
                                      norm_layer=norm_layer)

            self.merge1 = ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1, groups=1,
                                     norm_layer=norm_layer)
            self.merge2 = ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1, groups=1,
                                     norm_layer=norm_layer)
        else:
            self.output1 = ConvBNReLU(in_channel * 2, out_channel, kernel_size=1, stride=1,
                                      padding=0, groups=1, norm_layer=norm_layer)
            self.output2 = ConvBNReLU(in_channel * 4, out_channel, kernel_size=1, stride=1,
                                      padding=0, groups=1, norm_layer=norm_layer)
            self.output3 = ConvBNReLU(in_channel * 8, out_channel, kernel_size=1, stride=1,
                                      padding=0, groups=1, norm_layer=norm_layer)

            self.merge1 = ConvBNReLU(out_channel, out_channel, kernel_size=3, stride=1, padding=1,
                                     groups=1,
                                     norm_layer=norm_layer)
            self.merge2 = ConvBNReLU(out_channel, out_channel, kernel_size=3, stride=1, padding=1,
                                     groups=1,
                                     norm_layer=norm_layer)

    def construct(self, input1, input2, input3):
        """Forward pass."""
        output1 = self.output1(input1)
        output2 = self.output2(input2)
        output3 = self.output3(input3)

        up3 = ops.ResizeNearestNeighbor(
            [ops.Shape()(output2)[2], ops.Shape()(output2)[3]])(output3)
        output2 = up3 + output2
        output2 = self.merge2(output2)

        up2 = ops.ResizeNearestNeighbor(
            [ops.Shape()(output1)[2], ops.Shape()(output1)[3]])(output2)
        output1 = up2 + output1
        output1 = self.merge1(output1)

        return output1, output2, output3


class RetinaNet(nn.Cell):
    """
    RetinaNet network.
    """

    def __init__(self, phase='train', backbone=None, cfg=None):

        super(RetinaNet, self).__init__()
        self.phase = phase
        self.base = backbone
        if cfg is None:
            self.fpn = FPN()
            self.ssh1 = SSH(256, 256)
            self.ssh2 = SSH(256, 256)
            self.ssh3 = SSH(256, 256)
            self.class_head = self._make_class_head(
                fpn_num=3, inchannels=[256, 256, 256], anchor_num=[2, 2, 2])
            self.bbox_head = self._make_bbox_head(
                fpn_num=3, inchannels=[256, 256, 256], anchor_num=[2, 2, 2])
            self.landmark_head = self._make_landmark_head(
                fpn_num=3, inchannels=[256, 256, 256], anchor_num=[2, 2, 2])
        else:
            self.fpn = FPN(
                in_channel=cfg['in_channel'], out_channel=cfg['out_channel'])
            self.ssh1 = SSH(cfg['out_channel'],
                            cfg['out_channel'])
            self.ssh2 = SSH(cfg['out_channel'],
                            cfg['out_channel'])
            self.ssh3 = SSH(cfg['out_channel'],
                            cfg['out_channel'])
            self.class_head = self._make_class_head(fpn_num=3, inchannels=[cfg['out_channel'], cfg['out_channel'],
                                                                           cfg['out_channel']], anchor_num=[2, 2, 2])
            self.bbox_head = self._make_bbox_head(fpn_num=3, inchannels=[cfg['out_channel'], cfg['out_channel'],
                                                                         cfg['out_channel']], anchor_num=[2, 2, 2])
            self.landmark_head = self._make_landmark_head(fpn_num=3, inchannels=[cfg['out_channel'],
                                                                                 cfg['out_channel'],
                                                                                 cfg['out_channel']],
                                                          anchor_num=[2, 2, 2])

        self.cat = ops.Concat(axis=1)

    def _make_class_head(self, fpn_num, inchannels, anchor_num):
        """Construct class head of network."""
        classhead = nn.CellList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels[i], anchor_num[i]))
        return classhead

    def _make_bbox_head(self, fpn_num, inchannels, anchor_num):
        """Construct box head of network."""
        bboxhead = nn.CellList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels[i], anchor_num[i]))
        return bboxhead

    def _make_landmark_head(self, fpn_num, inchannels, anchor_num):
        """Construct landmark head of network."""
        landmarkhead = nn.CellList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels[i], anchor_num[i]))
        return landmarkhead

    def construct(self, inputs):
        """Forward pass."""
        f1, f2, f3 = self.base(inputs)
        f1, f2, f3 = self.fpn(f1, f2, f3)
        f1 = self.ssh1(f1)
        f2 = self.ssh2(f2)
        f3 = self.ssh3(f3)
        features = [f1, f2, f3]
        bbox = ()
        for i, feature in enumerate(features):
            bbox = bbox + (self.bbox_head[i](feature),)
        bbox_regressions = self.cat(bbox)
        cls = ()
        for i, feature in enumerate(features):
            cls = cls + (self.class_head[i](feature),)
        classifications = self.cat(cls)
        landm = ()
        for i, feature in enumerate(features):
            landm = landm + (self.landmark_head[i](feature),)
        ldm_regressions = self.cat(landm)
        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, ops.Softmax(-1)
                      (classifications), ldm_regressions)

        return output
