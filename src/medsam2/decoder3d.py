import torch
import torch.nn as nn


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, up=(1, 2, 2)):  # no depth change
        super().__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, 3, 1, 1)
        self.gn1 = nn.GroupNorm(8, out_c)
        self.act = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_c, out_c, 3, 1, 1)
        self.gn2 = nn.GroupNorm(8, out_c)
        self.up = nn.Upsample(scale_factor=up, mode="trilinear", align_corners=False)

    def forward(self, x):
        x = self.act(self.gn1(self.conv1(x)))
        x = self.act(self.gn2(self.conv2(x)))
        return self.up(x)


class Decoder3D(nn.Module):
    def __init__(self, num_classes=9, ch=256):
        super().__init__()
        self.b0 = UpBlock(ch, ch)  # 8×8 → 16×16
        self.b1 = UpBlock(ch, ch)  # 16×16 → 32×32
        self.b2 = UpBlock(ch, ch, up=(1, 4, 4))  # 32×32 → 128×128
        self.head = nn.Conv3d(ch, num_classes, 1)

    def forward(self, feats, atlas_features=None):  # feats = (f0,f1,f2)
        f0, f1, f2 = feats  # (B,C,128,32,32)… etc.
        if atlas_features is None:
            x = self.b0(f2)
        else:
            x = self.b0(f2 + atlas_features)  # (B,C,128,16,16)
        x = self.b1(f1 + x)  # (B,C,128,32,32)
        x = self.b2(f0 + x)  # (B,C,128,128,128)
        return self.head(x)


class Decoder3DModularized(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.decoder_background = Decoder3D(num_classes=1, ch=ch)
        self.decoder_hipp = Decoder3D(num_classes=2, ch=ch)
        self.decoder_extra = Decoder3D(num_classes=2, ch=ch)
        self.decoder_basal = Decoder3D(num_classes=4, ch=ch)

    def forward(self, feats):  # feats = (f0,f1,f2)
        out_background = self.decoder_background(feats)
        out_hipp = self.decoder_hipp(feats)
        out_extra = self.decoder_extra(feats)
        out_basal = self.decoder_basal(feats)
        out = torch.cat([out_background, out_hipp, out_extra, out_basal], dim=1)
        return out
