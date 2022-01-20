"""
modified from shufflenetv2 torchvision implementation
https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py
"""

import torch
import torch.nn as nn
from .commons import OutputLayer
from .atoms.TVConv import TVConv


__all__ = [
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5'
]


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, h_w, stride, **kwargs):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, h_w, kernel_size=3, stride=self.stride, padding=1, **kwargs),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, h_w, kernel_size=3, stride=self.stride, padding=1, **kwargs),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, h_w, kernel_size, stride=1, padding=0, bias=False, **kwargs):
        if kwargs["atom"]=='base':
            return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)
        elif kwargs["atom"]=='TVConv':
            return TVConv(i, h=h_w, w=h_w, stride=stride, **kwargs)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


def get_InvertedResidual(input_channels, output_channels, h_w, stride, **kwargs):
    return InvertedResidual(input_channels, output_channels, h_w, stride, **kwargs)


class ShuffleNetV2(nn.Module):
    def __init__(self, input_size, stages_repeats, h_ws, stages_out_channels, **kwargs):
        super(ShuffleNetV2, self).__init__()
        assert input_size[0] in [96, ]

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, h_w, output_channels in zip(
                stage_names, stages_repeats, h_ws, self._stage_out_channels[1:]):
            seq = [get_InvertedResidual(input_channels, output_channels, h_w, 2, **kwargs)]
            for i in range(repeats - 1):
                seq.append(get_InvertedResidual(output_channels, output_channels, h_w, 1, **kwargs))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        # self.fc = nn.Linear(output_channels, num_classes)

        # building classifier
        self.output_layer = OutputLayer(output_channels, drop_ratio=kwargs["drop_ratio"], feat_dim=512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        # x = x.mean([2, 3])  # globalpool
        # x = self.fc(x)

        x = self.output_layer(x)
        return x


def _shufflenetv2(input_size, stages_repeats, h_ws, stages_out_channels, **kwargs):
    model = ShuffleNetV2(input_size, stages_repeats, h_ws, stages_out_channels, **kwargs)
    return model


def shufflenet_v2_x0_5(input_size, **kwargs):
    return _shufflenetv2(input_size,
                         [4, 8, 4], [24, 12, 6], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(input_size, **kwargs):
    return _shufflenetv2(input_size,
                         [4, 8, 4], [24, 12, 6], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(input_size, **kwargs):
    return _shufflenetv2(input_size,
                         [4, 8, 4], [24, 12, 6], [24, 176, 352, 704, 1024], **kwargs)