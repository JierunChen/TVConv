"""
modified from mobilenetv2 torchvision implementation
https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenet.py
"""

from torch import nn
from .commons import OutputLayer
from .atoms.TVConv import TVConvInvertedResidual


__all__ = ['mobilenet_v2_x0_1', 'mobilenet_v2_x0_2', 'mobilenet_v2_x0_3',
           'mobilenet_v2_x0_5', 'mobilenet_v2_x1_0']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, h_w, expand_ratio, norm_layer=None, **kwargs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))

        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 input_size, embedding_size = 512,
                 width_mult=1.0, inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None,
                 **kwargs):
        """
        MobileNet V2 main class

        Args:
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()
        assert input_size[0] in [96, ]
        if block is None:
            if kwargs["atom"] == 'base':
                block = InvertedResidual
            elif kwargs["atom"] == 'TVConv':
                block = TVConvInvertedResidual
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 512

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, h_w
                [1, 16, 1, 1, 96],
                [6, 24, 2, 2, 48],         # might be removed
                [6, 32, 3, 2, 24],
                [6, 64, 4, 2, 12],
                [6, 96, 3, 1, 12],
                [6, 160, 3, 2, 6],
                [6, 320, 1, 1, 6]
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 5:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        # features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        features = [ConvBNReLU(3, input_channel, stride=1, norm_layer=norm_layer)]

        for t, c, n, s, h_w in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1

                features.append(block(input_channel, output_channel, stride, h_w, expand_ratio=t, norm_layer=norm_layer, **kwargs))
                input_channel = output_channel

        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.output_layer = OutputLayer(last_channel, drop_ratio=kwargs["drop_ratio"], feat_dim=embedding_size)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
                if isinstance(m, nn.BatchNorm2d) and kwargs["atom"] == 'invo':
                    pass
                else:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.output_layer(x)
        return x


def mobilenet_v2_x0_1(input_size, **kwargs):
    model = MobileNetV2(input_size, width_mult=0.1, **kwargs)
    return model


def mobilenet_v2_x0_2(input_size, **kwargs):
    model = MobileNetV2(input_size, width_mult=0.2, **kwargs)
    return model


def mobilenet_v2_x0_3(input_size, **kwargs):
    model = MobileNetV2(input_size, width_mult=0.3, **kwargs)
    return model


def mobilenet_v2_x0_5(input_size, **kwargs):
    model = MobileNetV2(input_size, width_mult=0.5, **kwargs)
    return model


def mobilenet_v2_x1_0(input_size, **kwargs):
    model = MobileNetV2(input_size, width_mult=1.0, **kwargs)
    return model