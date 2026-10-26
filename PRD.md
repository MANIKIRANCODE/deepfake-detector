# PRD: Deepfake Detection Web App

## 1. Problem & Goal
Give a user a simple web page to upload an image or short video and receive a real/fake verdict with a confidence score, using a face-crop classifier built on transfer learning.

## 2. Scope (v1)
- **In scope:** single-face images (jpg/png), short videos (mp4, <30s), one model (fine-tuned MobileNetV2 or EfficientNet-B0), local Streamlit UI, local training script, CPU or single-GPU training.
- **Out of scope (v1):** multi-face handling beyond "largest face", audio deepfake detection, cloud deployment, user accounts, mobile app.

## 3. Users
A single developer (you) building a portfolio/learning project; secondary audience is anyone reviewing the project (recruiters, course graders).

## 4. User Flow
User picks one of two modes on opening the app:

**Upload mode:**
1. User uploads an image or video file.
2. App detects the largest face (OpenCV Haar cascade), crops it.
3. App runs the crop(s) through the trained classifier.
4. For video: app samples ~16 evenly spaced frames, scores each, averages.
5. App displays: verdict (Genuine/Deepfake), confidence %, the cropped face(s), and a disclaimer.

**Live mode (simulating a live call/meeting check):**
1. User opens their webcam in-browser and clicks to capture a check whenever they want one (e.g. every 10-15s during a call).
2. Each snapshot goes through the same face-crop + classifier pipeline as a single image.
3. App shows the verdict for that check and appends it to a running session trend chart, so the user can see authenticity drift over a call rather than a single static result.

## 5. Functional Requirements
- FR1: Accept jpg/png/mp4 upload up to a set size limit (e.g. 50MB).
- FR2: Detect and crop the largest face; show an error if no face is found.
- FR3: Run inference with a locally saved model checkpoint (`.pt` file).
- FR4: For video, sample frames, score each, and show a per-frame score alongside the average.
- FR5: Display a clear disclaimer that the result is a probability, not proof.
- FR6: Provide a Live mode that opens the user's webcam and scores on-demand snapshots, showing a running trend across the session (v1: click-to-capture; continuous auto-sampling is a documented future upgrade).
- FR7: Use consistent terminology (Genuine / Deepfake) across both modes, matching the anti-spoofing framing of the project.

## 6. Non-Functional Requirements
- Single image inference: under 2 seconds on CPU.
- Video (16 frames): under 15 seconds on CPU, faster on GPU.
- Code must run on Windows with Python 3.10, using a local virtual environment.

## 7. Model & Data
- Architecture: MobileNetV2 or EfficientNet-B0 pretrained on ImageNet, fine-tuned with a new 2-class head.
- Dataset: Kaggle "140k Real and Fake Faces" (or equivalent), split train/val/test.
- Training: subset-first (1k/1k images) to validate the pipeline, then scale to full dataset.
- Target metric: >90% validation accuracy on held-out images from the same dataset; report accuracy separately on an out-of-distribution sample to show generalization limits honestly.

## 8. Risks & Known Limitations
- Face detector (Haar cascade) misses profile faces/occlusion — acceptable for v1, documented as a limitation.
- Model trained on one dataset will likely underperform on deepfakes made with unseen methods (documented, not solved, in v1).
- Not a forensic tool; results should not be treated as legal/definitive proof.
- v1 covers only the visual channel. The source presentation describes a full multimodal system (face + voice + lip-sync fusion via CNN+ViT and attention); v1 implements the face/CNN piece only and documents the rest as roadmap, not as delivered scope.
- Live mode in v1 is click-to-capture, not a fully automatic continuous stream; this is a known simplification, with `streamlit-webrtc` noted as the upgrade path.

## 9. Deliverables
- `src/` training pipeline (dataset loader, model, train script, eval script).
- `app/streamlit_app.py` web app for image + video inference.
- `README.md` with setup, training, and run instructions.
- Trained checkpoint produced by the user after running training locally (requires their GPU + downloaded dataset — not included, since training data/compute aren't available in this environment).

## 10. Out-of-the-box status
Everything except the actual training run ships ready-to-use: you run `train.py` locally against your downloaded dataset to produce the checkpoint the app needs.
