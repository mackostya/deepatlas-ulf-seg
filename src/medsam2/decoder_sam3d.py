"""
Decoder Blocks Taken from the Segment Anything Model (SAM) 3D implementation.
https://arxiv.org/html/2309.03493v4#bib.bib14
"""

import torch.nn as nn


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kSize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=True):
        super().__init__()

        self.conv_pred = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=kSize, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(in_channels // 2),
            nn.LeakyReLU(),
        )
        self.segmentation_head = nn.Conv3d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_pred(x)
        return self.segmentation_head(x)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kSize=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        upsample_factor=2,
        bias=True,
        mode="nearest",
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kSize, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=kSize, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(out_channels),
        )

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias),
            nn.InstanceNorm3d(out_channels),
        )

        self.leakyrelu = nn.LeakyReLU()

        self.up = nn.Upsample(scale_factor=(1, upsample_factor, upsample_factor), mode=mode)

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out += residual

        out = self.leakyrelu(out)
        out = self.up(out)
        return out


class DecoderSAM3D(nn.Module):
    def __init__(self, in_channels=256, hidden_channels=[256, 256, 256], num_classes=9):
        super().__init__()

        self.block0 = BasicBlock(in_channels, hidden_channels[0], upsample_factor=2)
        self.block1 = BasicBlock(in_channels, hidden_channels[1], upsample_factor=2)
        self.block2 = BasicBlock(in_channels, hidden_channels[2], upsample_factor=4)

        self.final = SegmentationHead(in_channels=hidden_channels[2], out_channels=num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        f0, f1, f2 = features  # Assuming features is a list of three tensors
        x = self.block0(f2)
        x = self.block1(self.relu(f1 + x))
        x = self.block2(self.relu(f0 + x))
        x = self.final(x)
        return x
