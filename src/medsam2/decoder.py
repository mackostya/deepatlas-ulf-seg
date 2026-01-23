import torch.nn as nn
import torch.nn.functional as F


class FPNDecoder2D(nn.Module):
    """
    A simple 2D FPN-style decoder that fuses MedSAM2's three backbone_fpn features:
      f0:  [BxD, 256, 32, 32]
      f1:  [BxD, 256, 16, 16]
      f2:  [BxD, 256,  8,  8]
    and produces a single-channel 128x128 map.
    """

    def __init__(self, feature_channels=[256, 256, 256], mid_channels=64, out_channels=9):
        """
        Args:
          feature_channels: list of ints, the channel dimension of [f0, f1, f2].
                            In MedSAM2, each is 256.
          mid_channels:      how many channels to reduce each lateral conv to (64 here).
          out_channels:      usually 1 (the 1-channel mask logit).
        """
        super().__init__()
        # 1x1 laterals: project each 256→ mid_channels
        self.lateral0 = nn.Conv2d(feature_channels[0], mid_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(feature_channels[1], mid_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(feature_channels[2], mid_channels, kernel_size=1)

        # after each add, do a conv3x3 for smoothing
        self.smooth1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.smooth0 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)

        # final 1x1 to map mid_channels→ out_channels (e.g. 1)
        self.final_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        # (Optionally, you could add BatchNorm+ReLU after each smooth conv. Here we’ll do ReLU only.)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats):
        """
        Args:
          feats: list of three tensors [f0, f1, f2], where
            f0: [BxD, 256, 32, 32]
            f1: [BxD, 256, 16, 16]
            f2: [BxD, 256,  8,  8]
        Returns:
          logit: [BxD, out_channels=1, 128, 128]
        """
        f0, f1, f2 = feats

        # 1) lateral conv on f2 (8x8)
        p2 = self.lateral2(f2)  # → [BxD, 64, 8, 8]
        p2_up = F.interpolate(p2, scale_factor=2, mode="bilinear", align_corners=False)
        #                       ↑  → [BxD, 64, 16, 16]

        # 2) lateral conv on f1 (16x16), then add p2_up
        l1 = self.lateral1(f1)  # → [BxD, 64, 16, 16]
        p1 = self.relu(self.smooth1(l1 + p2_up))  # → [BxD, 64, 16, 16]
        p1_up = F.interpolate(p1, scale_factor=2, mode="bilinear", align_corners=False)
        #                       ↑  → [BxD, 64, 32, 32]

        # 3) lateral conv on f0 (32x32), then add p1_up
        l0 = self.lateral0(f0)  # → [BxD, 64, 32, 32]
        p0 = self.relu(self.smooth0(l0 + p1_up))  # → [BxD, 64, 32, 32]

        # 4) upsample p0 from 32x32 → 128x128 (factor = 4)
        p0_up = F.interpolate(p0, scale_factor=4, mode="bilinear", align_corners=False)
        #           ↑ → [BxD, 64, 128, 128]

        # 5) final 1x1 conv → N-channel logit
        out = self.final_conv(p0_up)  # → [BxD, 9, 128, 128]
        return out
