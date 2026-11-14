"""CROMA encoders and a classifier for center-pixel patch labels."""

from pathlib import Path

import torch
from torch import nn

from model.pretrain_croma import CROMA


class PatchClassifier(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        # Convolve across the pooled vector's feature axis.
        self.conv = nn.Conv1d(1, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * dim, num_classes)

    def forward(self, tokens):
        pooled = tokens.mean(dim=1).unsqueeze(1)  # [B, 1, D]
        return self.fc(torch.nn.functional.gelu(self.conv(pooled)).flatten(1))


def build_encoders(config, image_size, radar_channels, optical_channels, device, root):
    args = dict(config)
    checkpoint = args.pop("pretrained_checkpoint", None)
    vit_patch = args["patch_size"]
    if vit_patch <= 0 or image_size % vit_patch:
        raise ValueError("dataset.patch_size must be divisible by croma.patch_size")
    model = CROMA(
        **args, num_patches=(image_size // vit_patch) ** 2,
        radar_channels=radar_channels, opt_channels=optical_channels,
        total_channels=radar_channels + optical_channels,
    )
    if checkpoint:
        path = root / Path(checkpoint).expanduser()
        state = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("model_state_dict", "model", "state_dict"):
            if key in state:
                state = state[key]
                break
        prefixes = ("radar_encoder.", "optical_encoder.", "cross_encoder.")
        state = {k.removeprefix("module."): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if k.startswith(prefixes)}
        missing, _ = model.load_state_dict(state, strict=False)
        if any(k.startswith(prefixes) for k in missing):
            raise ValueError(f"Checkpoint does not contain all three configured encoders: {path}")
    return (
        model.radar_encoder.to(device), model.optical_encoder.to(device),
        model.cross_encoder.to(device), model.attn_bias.to(device),
    )
