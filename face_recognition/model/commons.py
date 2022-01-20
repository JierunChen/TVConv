import torch.nn as nn
from torch.nn import BatchNorm1d, BatchNorm2d, Module


class Flatten(Module):

    def forward(self, input):
        return input.view(input.size(0), -1)


class OutputLayer(nn.Module):
    def __init__(self, channels, drop_ratio, feat_dim=512, size=6):
        super(OutputLayer, self).__init__()
        if drop_ratio>0:
            self.layer1 = nn.Sequential(
                BatchNorm2d(channels),
                nn.Dropout(drop_ratio)
            )
        else:
            self.layer1 = BatchNorm2d(channels)
        self.layer2 = nn.Sequential(
            nn.Linear(channels, feat_dim),
            BatchNorm1d(feat_dim)
        )
        self.pool = nn.AvgPool2d(size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # x = x.mean([2, 3])
        x = self.layer2(x)
        return x

