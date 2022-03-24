"""
    Common routines for models in PyTorch.
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Sequential):
    def __init__(self, in_planes, out_planes, h, w, kernel_size=3, stride=1, bias=False):
        padding = (kernel_size - 1) // 2

        super(_ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias),
            nn.LayerNorm([out_planes, h, w]),
            nn.ReLU(inplace=True)
        )


class TVConv(nn.Module):
    def __init__(self,
                 channels,
                 TVConv_k=3,
                 stride=1,
                 TVConv_posi_chans=4,
                 TVConv_inter_chans=16,
                 TVConv_inter_layers=1,
                 TVConv_Bias=False,
                 h=3,
                 w=3,
                 **kwargs):
        super(TVConv, self).__init__()

        self.register_buffer("TVConv_k", torch.as_tensor(TVConv_k))
        self.register_buffer("TVConv_k_square", torch.as_tensor(TVConv_k**2))
        self.register_buffer("stride", torch.as_tensor(stride))
        self.register_buffer("channels", torch.as_tensor(channels))
        self.register_buffer("h", torch.as_tensor(h))
        self.register_buffer("w", torch.as_tensor(w))

        self.bias_layers = None

        out_chans = self.TVConv_k_square * self.channels

        self.posi_map = nn.Parameter(torch.Tensor(1, TVConv_posi_chans, h, w))
        nn.init.ones_(self.posi_map)

        self.weight_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, out_chans, TVConv_inter_layers, h, w)
        if TVConv_Bias:
            self.bias_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, channels, TVConv_inter_layers, h, w)

        self.unfold = nn.Unfold(TVConv_k, 1, (TVConv_k-1)//2, stride)

    def _make_layers(self, in_chans, inter_chans, out_chans, num_inter_layers, h, w):
        layers = [_ConvBlock(in_chans, inter_chans, h, w, bias=False)]
        for i in range(num_inter_layers):
            layers.append(_ConvBlock(inter_chans, inter_chans, h, w, bias=False))
        layers.append(nn.Conv2d(
            in_channels=inter_chans,
            out_channels=out_chans,
            kernel_size=3,
            padding=1,
            bias=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        weight = self.weight_layers(self.posi_map)
        weight = weight.view(1, self.channels, self.TVConv_k_square, self.h, self.w)
        out = self.unfold(x).view(x.shape[0], self.channels, self.TVConv_k_square, self.h, self.w)
        out = (weight * out).sum(dim=2)

        if self.bias_layers is not None:
            bias = self.bias_layers(self.posi_map)
            out = out + bias

        return out


class TVConv_test(nn.Module):
    def __init__(self,
                 TVConv_k_square,
                 channels,
                 h,
                 w,
                 weight_maps,
                 unfold):
        super(TVConv_test, self).__init__()
        assert TVConv_k_square==9
        self.register_buffer("weight_maps", weight_maps.view(1, channels, 3, 3, h, w))
        # self.weight_maps = weight_maps.view(1, channels, 3, 3, h, w).detach().contiguous()
        self.register_buffer("c", torch.as_tensor(channels))
        self.register_buffer("h", torch.as_tensor(h))
        self.register_buffer("w", torch.as_tensor(w))
        if unfold.stride==2:
            h = 2 * h
            w = 2 * w
        h += 2
        w += 2
        self.strides = (channels*h*w, h*w, w, 1, w, unfold.stride)

    def forward(self, x):
        # x = nn.functional.pad(x, (1, 1, 1, 1), "constant", 0).contiguous()
        x = nn.functional.pad(x, (1, 1, 1, 1), "constant", 0)
        out = torch.as_strided(x, (x.shape[0], self.c, 3, 3, self.h, self.w), self.strides)
        out = (self.weight_maps * out).sum(dim=(2, 3))

        return out


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


class TVConvBNReLU(nn.Sequential):
    def __init__(self, planes, h_w, stride=1, norm_layer=None, **kwargs):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(TVConvBNReLU, self).__init__(
            TVConv(planes, h=h_w, w=h_w, stride=stride, **kwargs),
            norm_layer(planes),
            nn.ReLU6(inplace=True)
        )


class TVConvInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, h_w, expand_ratio, norm_layer=None, **kwargs):
        super(TVConvInvertedResidual, self).__init__()
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
            TVConvBNReLU(hidden_dim, h_w, stride=stride, norm_layer=norm_layer, **kwargs),
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


def model_transform_for_test(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            model_transform_for_test(module)

        if isinstance(module, TVConv):
            new_TVConv = TVConv_test(
                module.TVConv_k_square.detach().item(),
                module.channels.detach().item(),
                module.h.detach().item(),
                module.w.detach().item(),
                module.weight_layers(module.posi_map).detach(),
                module.unfold)
            setattr(model, n, new_TVConv)

    return model


if __name__ == '__main__':
    torch.manual_seed(0)

    c = 3
    h = 96
    w = 96

    data = torch.randn(1, c, h, w)

    with torch.no_grad():
        model1 = TVConv(
            channels=c,
            TVConv_k=3,
            stride=1,
            TVConv_posi_chans=4,
            TVConv_inter_chans=16,
            TVConv_inter_layers=1,
            TVConv_Bias=False,
            h=h,
            w=w
        )
        model2 = TVConv_test(
            model1.TVConv_k_square.detach().item(),
            model1.channels.detach().item(),
            model1.h.detach().item(),
            model1.w.detach().item(),
            model1.weight_layers(model1.posi_map).detach(),
            model1.unfold)


        out1 = model1(data)
        out2 = model2(data)
        print(torch.eq(out1, out2))
        print(torch.all(torch.eq(out1, out2)))

        # with torch.autograd.profiler.profile() as prof:
        #     for _ in range(100):  # any normal python code, really!
        #         # model1(data)
        #         model2(data)
        # print(prof.key_averages().table(sort_by="self_cpu_time_total"))


