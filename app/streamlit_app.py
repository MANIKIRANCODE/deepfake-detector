"""
Real-time deepfake / anti-spoofing checker -- web app.

Two modes, matching the "live authentication" framing of the project:
  - Upload: analyze a face image or a recorded video file.
  - Live check: open your webcam and snap periodic checks, the way a live
    meeting or exam proctoring system would sample frames during a call.

Run from the project root (one level above this file's folder), with your
virtual environment activated:

    streamlit run app/streamlit_app.py

Requires a trained checkpoint at weights/model.pt (produced by src/train.py).
"""

import os
import sys
import tempfile
from collections import deque

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Make src/ importable regardless of where streamlit is launched from.
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from face_utils import FaceCropper
from model import load_checkpoint
from infer import score_image, score_video
from lipsync import check_lip_sync

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "model.pt")
LIVE_HISTORY_MAXLEN = 30  # how many past live checks to keep for the trend chart


@st.cache_resource
def load_resources():
    cropper = FaceCropper()
    if not os.path.exists(CHECKPOINT_PATH):
        return cropper, None
    model = load_checkpoint(CHECKPOINT_PATH, device="cpu")
    return cropper, model


def verdict_label(fake_prob: float) -> str:
    # Labels match the PPT's terminology: the system outputs Genuine or Deepfake.
    return "DEEPFAKE" if fake_prob >= 0.5 else "GENUINE"


def render_verdict(fake_prob: float):
    label = verdict_label(fake_prob)
    confidence = fake_prob if label == "DEEPFAKE" else 1 - fake_prob
    color = "red" if label == "DEEPFAKE" else "green"
    st.markdown(
        f"### Verdict: :{color}[{label}]  \nConfidence: {confidence * 100:.1f}%"
    )
    st.progress(fake_prob, text=f"Deepfake probability: {fake_prob * 100:.1f}%")


def process_face_image(cropper, model, rgb: np.ndarray, source_caption: str):
    """Shared logic for scoring a single RGB image, used by both Upload and Live modes."""
    crop = cropper.crop_largest_face(rgb)
    if crop is None:
        st.warning("No face was detected in this frame. Try a clearer, front-facing view.")
        return None

    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb, caption=source_caption, use_container_width=True)
    with col2:
        st.image(crop, caption="Detected face (what the model sees)", use_container_width=True)

    fake_prob = score_image(model, crop, device="cpu")
    render_verdict(fake_prob)
    return fake_prob


def run_upload_mode(cropper, model):
    file = st.file_uploader(
        "Upload an image or video",
        type=["jpg", "jpeg", "png", "mp4", "mov", "avi"],
    )

    if file is None:
        st.info("Waiting for a file to analyze.")
        return

    is_video = file.type.startswith("video")

    if is_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
            tmp.write(file.read())
            video_path = tmp.name

        st.video(video_path)
        try:
            with st.spinner("Sampling frames and scoring..."):
                try:
                    result = score_video(model, video_path, cropper, device="cpu")
                except ValueError as e:
                    st.warning(str(e))
                    return

            render_verdict(result["average_score"])
            st.caption(
                f"Scored {result['frames_with_face']} of {result['frames_sampled']} "
                "sampled frames (others had no detectable face)."
            )
            with st.expander("Per-frame scores"):
                st.bar_chart(result["frame_scores"])

            st.subheader("Lip-sync check")
            st.caption(
                "Heuristic signal, not a trained AI model -- compares mouth-motion "
                "timing to audio loudness timing. Treat as one extra clue, not a verdict."
            )
            with st.spinner("Checking lip-sync..."):
                try:
                    sync_result = check_lip_sync(video_path)
                except Exception as e:
                    sync_result = {"available": False, "reason": f"Lip-sync check failed: {e}"}

            if not sync_result["available"]:
                st.info(sync_result["reason"])
            else:
                score = sync_result["sync_score"]
                st.write(f"**{sync_result['verdict']}**")
                st.progress(
                    max(0.0, min(1.0, (score + 1) / 2)),
                    text=f"Sync score: {score:.2f} (based on {sync_result['frames_used']} frames)",
                )
        finally:
            os.unlink(video_path)

    else:
        image = Image.open(file).convert("RGB")
        rgb = np.array(image)
        process_face_image(cropper, model, rgb, source_caption="Uploaded image")


def run_live_mode(cropper, model):
    """c
    Live/meeting-style check: the user snaps a webcam photo whenever they want
    a check done (e.g. every 10-15 seconds during a call), instead of
    uploading a pre-recorded file. Each snapshot is scored immediately and
    added to a running trend chart, the way a liveness/anti-spoofing system
    would sample frames continuously during a session.

    This uses Streamlit's built-in camera widget, which opens the browser's
    webcam -- no extra packages or local camera drivers to configure. For a
    fully automatic, no-click continuous stream (closer to a production
    system), see the "streamlit-webrtc" suggestion in the README.
    """
    st.info(
        "Simulates checking identity during a live call or meeting: click "
        "**Take Photo** below whenever you want a fresh check. Keep snapping "
        "every so often to see the trend update, the way a real-time monitor "
        "would sample frames continuously."
    )

    if "live_history" not in st.session_state:
        st.session_state.live_history = deque(maxlen=LIVE_HISTORY_MAXLEN)

    snapshot = st.camera_input("Live camera check")

    if snapshot is not None:
        image = Image.open(snapshot).convert("RGB")
        rgb = np.array(image)
        fake_prob = process_face_image(cropper, model, rgb, source_caption="Live snapshot")
        if fake_prob is not None:
            st.session_state.live_history.append(fake_prob)

    if st.session_state.live_history:
        st.subheader("Session trend")
        st.line_chart(list(st.session_state.live_history))
        latest_avg = sum(st.session_state.live_history) / len(st.session_state.live_history)
        st.caption(
            f"Average deepfake probability across {len(st.session_state.live_history)} "
            f"check(s) this session: {latest_avg * 100:.1f}%"
        )
        if st.button("Clear session history"):
            st.session_state.live_history.clear()
            st.rerun()


def main():
    st.set_page_config(page_title="Real-Time Deepfake & Anti-Spoofing Check", page_icon="🕵️")
    st.title("🕵️ Real-Time Deepfake & Anti-Spoofing Check")
    st.caption(
        "A live-authentication style demo: verify whether a face feed is genuine "
        "or synthetic, either from an uploaded file or a live webcam check."
    )

    cropper, model = load_resources()

    if model is None:
        st.error(
            f"No trained model found at `{CHECKPOINT_PATH}`. "
            "Run `src/train.py` first to produce a checkpoint, then restart this app."
        )
        st.stop()

    mode = st.radio(
        "Choose how to check a face",
        ["📁 Upload (image or recorded video)", "🔴 Live check (webcam, like a live call)"],
        horizontal=True,
    )

    st.divider()

    if mode.startswith("📁"):
        run_upload_mode(cropper, model)
    else:
        run_live_mode(cropper, model)

    st.divider()
    with st.expander("About this system / limitations"):
        st.markdown(
            "This demo implements the **visual channel** of a real-time "
            "anti-spoofing pipeline: face detection + a CNN classifier scoring "
            "genuine vs. deepfake, run per-frame so it can be used live.\n\n"
            "**Not yet implemented (roadmap):** the voice/speaker-embedding "
            "channel, lip-sync / audio-visual consistency check, and the "
            "attention-based fusion of modalities into one authenticity score. "
            "Those would turn this into the full multimodal system."
        )
    st.caption(
        "⚠️ This tool outputs a statistical probability based on patterns in its "
        "training data. It is not forensic proof and can be wrong, especially on "
        "deepfake methods or face types it wasn't trained on. Do not use it as the "
        "sole basis for serious decisions."
    )


if __name__ == "__main__":
    main()
