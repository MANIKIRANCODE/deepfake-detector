"""
Lip-sync consistency check for uploaded videos.

IMPORTANT -- WHAT THIS IS AND ISN'T:
This is a heuristic correlation check, not a trained AI model. It measures
how much motion happens in the lower part of the face (a proxy for mouth
movement) and checks whether that motion lines up in time with the loudness
of the audio. Real lip-sync deepfake detectors (like the SyncNet approach
your presentation references) are trained neural networks learned from
thousands of hours of video -- building one from scratch is a much bigger
project. This heuristic is a reasonable first signal -- a face with lots of
audio but little mouth-area motion, or motion with no timing relationship to
the audio at all, is suspicious -- but it can be wrong: head movement,
lighting changes, background music, or a naturally still speaker can all
throw it off. Treat its output as one extra clue, not a verdict.

HOW IT WORKS
1. Sample frames evenly through the video.
2. For each sampled frame, find the face (reusing the same Haar-cascade
   detector as the rest of the project) and crop the lower third of it,
   where the mouth sits.
3. Measure how much that lower-face region changes from one sampled frame
   to the next (mean pixel difference) -- more change means more mouth/jaw
   motion.
4. Extract the audio track and compute its loudness (RMS energy) between
   those same sampled times.
5. Compare the two time series: in properly-synced speech, mouth motion and
   audio loudness tend to rise and fall together.
"""

import os

import cv2
import numpy as np

from face_utils import FaceCropper


def _sample_frame_times(video_path: str, num_samples: int):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= num_samples:
        indices = list(range(total_frames))
    else:
        step = total_frames / num_samples
        indices = [int(i * step) for i in range(num_samples)]

    frames_and_times = []
    idx_set = set(indices)
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx in idx_set:
            t = frame_idx / fps
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_and_times.append((t, rgb))
        frame_idx += 1
    cap.release()
    return frames_and_times


def _mouth_region(rgb_frame: np.ndarray, cropper: FaceCropper, size=(60, 30)):
    """Returns a small, fixed-size grayscale crop of the lower third of the
    largest detected face (an approximation of the mouth area), or None if
    no face was found."""
    boxes = cropper.find_faces(rgb_frame)
    if not boxes:
        return None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])

    # Lower third of the face box, middle 70% of its width.
    mouth_y0 = y + int(h * 0.62)
    mouth_y1 = y + int(h * 0.95)
    mouth_x0 = x + int(w * 0.15)
    mouth_x1 = x + int(w * 0.85)

    mouth_x0, mouth_y0 = max(0, mouth_x0), max(0, mouth_y0)
    mouth_x1 = min(rgb_frame.shape[1], mouth_x1)
    mouth_y1 = min(rgb_frame.shape[0], mouth_y1)

    if mouth_x1 <= mouth_x0 or mouth_y1 <= mouth_y0:
        return None

    crop = rgb_frame[mouth_y0:mouth_y1, mouth_x0:mouth_x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, size).astype(np.float32)


def _audio_energy_at_times(video_path: str, times, window_ms: float = 150.0):
    """Returns an RMS-loudness value at each given time, or None if the
    video has no audio track."""
    try:
        from moviepy import VideoFileClip  # moviepy v2 import path
    except ImportError:
        from moviepy.editor import VideoFileClip  # moviepy v1 fallback

    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        return None

    sr = 16000
    samples = clip.audio.to_soundarray(fps=sr)
    clip.close()
    if samples.ndim == 2:
        samples = samples.mean(axis=1)

    half_window = int(sr * window_ms / 1000 / 2)
    energies = []
    for t in times:
        center = int(t * sr)
        lo = max(0, center - half_window)
        hi = min(len(samples), center + half_window)
        window = samples[lo:hi]
        rms = float(np.sqrt(np.mean(window ** 2))) if len(window) > 0 else 0.0
        energies.append(rms)
    return np.array(energies)


def _zscore(arr: np.ndarray) -> np.ndarray:
    std = arr.std()
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def check_lip_sync(video_path: str, num_samples: int = 30) -> dict:
    """
    Returns a dict:
        {
          "available": bool,       # False if no face/audio usable
          "reason": str,           # explains why, if not available
          "sync_score": float,     # roughly -1 to 1; higher = better sync
          "verdict": str,          # short human-readable summary
          "frames_used": int,
        }
    """
    frames_and_times = _sample_frame_times(video_path, num_samples)
    if not frames_and_times:
        return {"available": False, "reason": "Could not read any frames from the video."}

    cropper = FaceCropper()

    mouth_crops = []
    crop_times = []
    for t, rgb in frames_and_times:
        region = _mouth_region(rgb, cropper)
        if region is not None:
            mouth_crops.append(region)
            crop_times.append(t)

    if len(mouth_crops) < 6:
        return {
            "available": False,
            "reason": "Could not reliably detect a face across enough frames "
                      "(try a video with a clearer, front-facing view).",
        }

    # Motion between consecutive kept frames, as a proxy for mouth movement.
    motion = [
        float(np.mean(np.abs(mouth_crops[i] - mouth_crops[i - 1])))
        for i in range(1, len(mouth_crops))
    ]
    # Pair each motion value with the midpoint time between the two frames
    # it was computed from.
    motion_times = [
        (crop_times[i] + crop_times[i - 1]) / 2 for i in range(1, len(crop_times))
    ]

    audio_energy = _audio_energy_at_times(video_path, motion_times)
    if audio_energy is None:
        return {"available": False, "reason": "This video has no audio track to compare against."}

    motion_z = _zscore(np.array(motion))
    audio_z = _zscore(audio_energy)

    # Correlation at a few small time shifts, allowing for a little natural
    # audio/video offset rather than requiring perfect frame alignment.
    best_corr = -1.0
    for shift in range(-2, 3):
        if shift < 0:
            a, b = motion_z[-shift:], audio_z[: len(audio_z) + shift]
        elif shift > 0:
            a, b = motion_z[: len(motion_z) - shift], audio_z[shift:]
        else:
            a, b = motion_z, audio_z
        if len(a) < 3:
            continue
        corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
        if not np.isnan(corr):
            best_corr = max(best_corr, corr)

    if best_corr > 0.4:
        verdict = "Mouth-area motion lines up well with the audio."
    elif best_corr > 0.15:
        verdict = "Mouth-area motion loosely matches the audio -- not a strong signal either way."
    else:
        verdict = "Mouth-area motion does NOT line up well with the audio -- worth a closer look."

    return {
        "available": True,
        "sync_score": best_corr,
        "verdict": verdict,
        "frames_used": len(mouth_crops),
    }
