import torch

import torch.nn as nn
import torch.nn.functional as F


# The Atlas3DEncoder is modified to output a tensor of shape (batch_size, 128, 16, 16)
class Atlas3DEncoder(nn.Module):
    def __init__(self, in_channels=2, base_channels=64, out_channels=256):
        super(Atlas3DEncoder, self).__init__()
        # Input: (B, 1, 128, 128, 128)
        self.conv1 = nn.Conv3d(
            in_channels, base_channels, kernel_size=3, stride=1, padding=1
        )  # (B, 64, 128, 128, 128)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.conv2 = nn.Conv3d(
            base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1
        )  # (B, 128, 64, 64, 64)
        self.bn2 = nn.BatchNorm3d(base_channels * 2)
        self.conv3 = nn.Conv3d(
            base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1
        )  # (B, 256, 32, 32, 32)
        self.bn3 = nn.BatchNorm3d(base_channels * 4)
        self.conv4 = nn.Conv3d(
            base_channels * 4, out_channels, kernel_size=3, stride=4, padding=1
        )  # (B, 256, 8, 8, 8)
        self.bn4 = nn.BatchNorm3d(out_channels)

    def forward(self, x, interpolate_to: int = 128):
        x = x.to(dtype=torch.float32)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        # x shape: (batch, out_channels, D, H, W) == (1, 256, 8, 8, 8) for (128,128,128) input
        x = F.interpolate(x, size=(interpolate_to, x.size(3), x.size(4)), mode="trilinear", align_corners=False)
        return x
