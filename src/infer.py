"""
Inference helpers used by the Streamlit app.

Two entry points:
- score_image(model, rgb_crop): score a single already-cropped face image.
- score_video(model, video_path, cropper): sample frames from a video,
  crop the face in each, score each, and return the per-frame scores
  plus their average.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import IMAGE_SIZE, CLASS_NAMES
from torchvision import transforms

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

FAKE_INDEX = CLASS_NAMES.index("fake")


def score_image(model, rgb_crop: np.ndarray, device: str = "cpu") -> float:
    """
    rgb_crop: HxWx3 uint8 RGB array of a cropped face.
    Returns a float in [0, 1]: probability the face is fake.
    """
    pil_image = Image.fromarray(rgb_crop)
    tensor = _transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)

    return probs[0, FAKE_INDEX].item()


def sample_frame_indices(total_frames: int, num_samples: int = 16):
    """Evenly spaced frame indices across the video, including first and last."""
    if total_frames <= num_samples:
        return list(range(total_frames))
    step = total_frames / num_samples
    return [int(i * step) for i in range(num_samples)]


def score_video(model, video_path: str, cropper, device: str = "cpu", num_samples: int = 16):
    """
    Returns a dict:
        {
          "frame_scores": [float, ...],   # fake-probability per sampled frame with a face
          "average_score": float,         # mean of frame_scores
          "frames_with_face": int,
          "frames_sampled": int,
        }
    Raises ValueError if no face was found in any sampled frame.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(sample_frame_indices(total_frames, num_samples))

    frame_scores = []
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx in indices:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            crop = cropper.crop_largest_face(rgb)
            if crop is not None:
                frame_scores.append(score_image(model, crop, device))
        frame_idx += 1
    cap.release()

    if not frame_scores:
        raise ValueError(
            "No face was detected in any sampled frame. Try a video with a "
            "clearer, more front-facing view."
        )

    return {
        "frame_scores": frame_scores,
        "average_score": sum(frame_scores) / len(frame_scores),
        "frames_with_face": len(frame_scores),
        "frames_sampled": len(indices),
    }
