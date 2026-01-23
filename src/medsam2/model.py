import os
import torch
from sam2.build_sam import build_sam2_video_predictor_npz
import torch.nn as nn
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir
from src.medsam2.decoder import FPNDecoder2D
from src.medsam2.decoder_sam3d import DecoderSAM3D
from src.medsam2.decoder3d import Decoder3D, Decoder3DModularized
from src.medsam2.atlas_encoder import Atlas3DEncoder


class MedSam2VolumetricSegmentor(nn.Module):
    def __init__(self):
        super().__init__()
        config_dir = os.getcwd() + "/configs/medsam2/"
        model_cfg = "sam2.1_hiera_t512.yaml"
        checkpoint = "checkpoints/MedSAM2_latest.pt"
        GlobalHydra.instance().clear()  # Clear any previous Hydra state
        initialize_config_dir(config_dir=config_dir, version_base="1.2")
        predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)
        self.image_encoder = predictor.image_encoder  # Placeholder for image encoder

        # freeze all parameters in image encoder
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False
        self.decoder = FPNDecoder2D()

    def forward(self, vol: torch.Tensor, atlas: torch.Tensor = None):
        B, _, D, H, W = vol.shape
        vol = vol.movedim(2, 1)  # [B, 3, D, H, W]
        vol = vol.repeat(1, 1, 3, 1, 1)
        vol = vol.view(B * D, 3, H, W)
        mean = torch.tensor([0.485, 0.456, 0.406], device=vol.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=vol.device).view(1, 3, 1, 1)
        vol = (vol - mean) / std
        out = self.image_encoder(vol)
        feats = out["backbone_fpn"]
        logits_flat = self.decoder(feats)
        logits = logits_flat.view(B, D, 9, H, W)
        logits = logits.movedim(2, 1)
        return logits


class MedSam2VolumetricSegmentor3D(nn.Module):
    def __init__(self, decoder_type="3d"):
        super().__init__()
        config_dir = os.getcwd() + "/configs/medsam2/"
        model_cfg = "sam2.1_hiera_t512.yaml"
        checkpoint = "checkpoints/MedSAM2_latest.pt"
        GlobalHydra.instance().clear()  # Clear any previous Hydra state
        initialize_config_dir(config_dir=config_dir, version_base="1.2")
        predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)
        self.image_encoder = predictor.image_encoder  # Placeholder for image encoder
        self.medsam2_dim = 256

        # freeze all parameters in image encoder
        for _, param in self.image_encoder.named_parameters():
            param.requires_grad = False

        # for name, param in self.image_encoder.neck.named_parameters():
        #     param.requires_grad = True
        # for name, param in self.image_encoder.trunk.blocks[11].named_parameters():
        #     param.requires_grad = True
        self.atlas_encoder = None
        if decoder_type == "3d":
            self.decoder = Decoder3D()
        elif decoder_type == "3d_atlas":
            self.decoder = Decoder3D()
            self.atlas_encoder = Atlas3DEncoder()
        elif decoder_type == "3d_mod":
            self.decoder = Decoder3DModularized()
        elif decoder_type == "sam3d":
            self.decoder = DecoderSAM3D()
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}. Supported types are 'sam3d' and '3d'.")

    def forward(self, vol: torch.Tensor, atlas: torch.Tensor = None):
        B, _, D, H, W = vol.shape
        vol = vol.movedim(2, 1)  # [B, 1, D, H, W] -> [B, D, 1, H, W]

        # if atlas is not None:
        #     atlas = atlas.unsqueeze(1).movedim(2, 1)
        #     vol = vol.repeat(1, 1, 2, 1, 1)
        #     vol = torch.cat((vol, atlas), dim=2)  # [B, D, 2+1, H, W]
        # else:
        vol = vol.repeat(1, 1, 3, 1, 1)
        vol = vol.view(B * D, 3, H, W).contiguous(memory_format=torch.channels_last)
        mean = torch.tensor([0.485, 0.456, 0.406], device=vol.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=vol.device).view(1, 3, 1, 1)
        vol = (vol - mean) / std

        out = self.image_encoder(vol)
        if self.atlas_encoder is not None and atlas is not None:
            atlas_features = self.atlas_encoder(atlas, interpolate_to=D)
        else:
            atlas_features = None
        feats = []
        for el in out["backbone_fpn"]:
            el = el.view(B, D, self.medsam2_dim, el.shape[-2], el.shape[-1])
            feats.append(el.movedim(1, 2))
        logits = self.decoder(feats, atlas_features)
        return logits
