"""
Model definition.

Uses a MobileNetV2 pretrained on ImageNet, with its final classifier layer
replaced by a new one that outputs 2 classes: real (0) and fake (1).
This is "transfer learning": the early layers already know how to see edges,
textures, and shapes, so we only need to retrain the last layer(s) to answer
our specific question.
"""

import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 2
CLASS_NAMES = ["real", "fake"]
IMAGE_SIZE = 224  # MobileNetV2's expected input size


def build_model(pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """
    pretrained: load ImageNet weights (recommended -- this is the whole point
        of transfer learning).
    freeze_backbone: if True, only the new final layer gets trained at first.
        This trains fast and avoids wrecking the pretrained features on a
        small dataset. You can set this to False later for "fine-tuning" once
        the head is already decent, using a lower learning rate.
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # MobileNetV2's classifier is Sequential(Dropout, Linear(1280 -> 1000)).
    # Replace the last Linear so it outputs NUM_CLASSES instead of 1000.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

    return model


def load_checkpoint(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Load a trained model from a .pt file saved by train.py."""
    model = build_model(pretrained=False, freeze_backbone=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
