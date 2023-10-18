import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops



def _last_conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod='same', pad=0):
    in_channels = in_channel
    out_channels = in_channel
    depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same',
                               padding=pad, group=in_channels)
    conv = nn.Conv2d(in_channel, out_channel, kernel_size=1,
                     stride=1, padding=0, pad_mode='same', has_bias=True)
    bn = nn.BatchNorm2d(in_channel, eps=1e-3, momentum=0.97,
                        gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

    return nn.SequentialCell([depthwise_conv, bn, nn.ReLU6(), conv])


def _make_layer(channels):
    in_channels = channels[0]
    layers = []
    for out_channels in channels[1:]:
        layers.append(nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels, kernel_size=3))
        layers.append(nn.ReLU())
        in_channels = out_channels
    return nn.SequentialCell(layers)


class Vgg16(nn.Cell):
    """VGG16 module."""

    def __init__(self):
        super(Vgg16, self).__init__()
        self.b1 = _make_layer([3, 64, 64])
        self.b2 = _make_layer([64, 128, 128])
        self.b3 = _make_layer([128, 256, 256, 256])
        self.b4 = _make_layer([256, 512, 512, 512])
        self.b5 = _make_layer([512, 512, 512, 512])

        self.m1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
        self.m2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
        self.m3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
        self.m4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
        self.m5 = nn.MaxPool2d(kernel_size=3, stride=1, pad_mode='SAME')

    def construct(self, x):
        # block1
        x = self.b1(x)
        x = self.m1(x)

        # block2
        x = self.b2(x)
        x = self.m2(x)

        # block3
        x = self.b3(x)
        x = self.m3(x)

        # block4
        x = self.b4(x)
        block4 = x
        x = self.m4(x)

        # block5
        x = self.b5(x)
        x = self.m5(x)

        return block4, x


class FlattenConcat(nn.Cell):
    """FlattenConcat module."""

    def __init__(self):
        super(FlattenConcat, self).__init__()
        self.num_ssd_boxes = 8732

    def construct(self, inputs):
        output = ()
        batch_size = ops.shape(inputs[0])[0]
        for x in inputs:
            x = ops.transpose(x, (0, 2, 3, 1))
            output += (ops.reshape(x, (batch_size, -1)),)
        res = ops.concat(output, axis=1)
        return ops.reshape(res, (batch_size, self.num_ssd_boxes, -1))


class MultiBox(nn.Cell):
    """
    Multibox conv layers. Each multibox layer contains class conf scores and localization predictions.
    """

    def __init__(self):
        super(MultiBox, self).__init__()
        num_classes = 81
        out_channels = [512, 1024, 512, 256, 256, 256]
        num_default = [4, 6, 6, 6, 4, 4]

        loc_layers = []
        cls_layers = []
        for k, out_channel in enumerate(out_channels):
            loc_layers += [_last_conv2d(out_channel, 4 * num_default[k],
                                        kernel_size=3, stride=1, pad_mod='same', pad=0)]
            cls_layers += [_last_conv2d(out_channel, num_classes * num_default[k],
                                        kernel_size=3, stride=1, pad_mod='same', pad=0)]

        self.multi_loc_layers = nn.CellList(loc_layers)
        self.multi_cls_layers = nn.CellList(cls_layers)
        self.flatten_concat = FlattenConcat()

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        for i in range(len(self.multi_loc_layers)):
            loc_outputs += (self.multi_loc_layers[i](inputs[i]),)
            cls_outputs += (self.multi_cls_layers[i](inputs[i]),)
        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)


class SSD300Vgg16(nn.Cell):
    """SSD300Vgg16 module."""

    def __init__(self):
        super(SSD300Vgg16, self).__init__()

        # VGG16 backbone: block1~5
        self.backbone = Vgg_model.Vgg16()

        # SSD blocks: block6~7
        self.b6_1 = nn.Conv2d(in_channels=512, out_channels=1024,
                              kernel_size=3, padding=6, dilation=6, pad_mode='pad')
        self.b6_2 = nn.Dropout(p=0.5)

        self.b7_1 = nn.Conv2d(
            in_channels=1024, out_channels=1024, kernel_size=1)
        self.b7_2 = nn.Dropout(p=0.5)

        # Extra Feature Layers: block8~11
        self.b8_1 = nn.Conv2d(in_channels=1024, out_channels=256,
                              kernel_size=1, padding=1, pad_mode='pad')
        self.b8_2 = nn.Conv2d(in_channels=256, out_channels=512,
                              kernel_size=3, stride=2, pad_mode='valid')

        self.b9_1 = nn.Conv2d(in_channels=512, out_channels=128,
                              kernel_size=1, padding=1, pad_mode='pad')
        self.b9_2 = nn.Conv2d(in_channels=128, out_channels=256,
                              kernel_size=3, stride=2, pad_mode='valid')

        self.b10_1 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=1)
        self.b10_2 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, pad_mode='valid')

        self.b11_1 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=1)
        self.b11_2 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, pad_mode='valid')

        # boxes
        self.multi_box = MultiBox()

    def construct(self, x):
        # VGG16 backbone: block1~5
        block4, x = self.backbone(x)

        # SSD blocks: block6~7
        x = self.b6_1(x)  # 1024
        x = self.b6_2(x)

        x = self.b7_1(x)  # 1024
        x = self.b7_2(x)
        block7 = x

        # Extra Feature Layers: block8~11
        x = self.b8_1(x)  # 256
        x = self.b8_2(x)  # 512
        block8 = x

        x = self.b9_1(x)  # 128
        x = self.b9_2(x)  # 256
        block9 = x

        x = self.b10_1(x)  # 128
        x = self.b10_2(x)  # 256
        block10 = x

        x = self.b11_1(x)  # 128
        x = self.b11_2(x)  # 256
        block11 = x

        # boxes
        multi_feature = (block4, block7, block8, block9, block10, block11)
        pred_loc, pred_label = self.multi_box(multi_feature)
        if not self.training:
            pred_label = ops.sigmoid(pred_label)
        pred_loc = pred_loc.astype(ms.float32)
        pred_label = pred_label.astype(ms.float32)
        return pred_loc, pred_label
