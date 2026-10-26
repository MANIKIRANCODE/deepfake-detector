# Deepfake Detector (Web App)

A beginner-friendly deepfake detector: upload a face image or short video,
get a real/fake verdict with a confidence score. Built with transfer learning
(MobileNetV2 fine-tuned on real vs. fake faces) + OpenCV face cropping +
a Streamlit UI.

See `PRD.md` for the full project scope and goals.

## 1. Setup

```
# from inside this project folder, with your virtual environment activated
pip install -r requirements.txt
```

## 2. Get data

Download the Kaggle "140k Real and Fake Faces" dataset (or similar), which
already comes organized like this:

```
data/
  train/
    real/
    fake/
  valid/
    real/
    fake/
  test/
    real/
    fake/
```

Unzip it so this structure sits at `data/` in the project root.

**Start small.** Before training on the full dataset, copy ~1,000 real and
~1,000 fake images per split into a `data_small/` folder with the same
structure, and train on that first to confirm everything works end-to-end.

If your own images are full photos rather than pre-cropped faces, run:

```
cd scripts
python prepare_dataset.py --in_dir path/to/raw/real --out_dir ../data/train/real
python prepare_dataset.py --in_dir path/to/raw/fake --out_dir ../data/train/fake
# repeat for valid/ and test/
```

## 3. Train

```
cd src
python train.py --data_dir ../data_small --epochs 5 --out ../weights/model.pt
```

Watch the printed `train_acc`/`val_acc` each epoch. Once this small run
works and improves, rerun pointing `--data_dir` at your full `data/` folder
and increase `--epochs` (10-15 is a reasonable start).

The script saves the best-so-far model to `weights/model.pt` automatically.

## 4. Run the web app

```
# from the project root
streamlit run app/streamlit_app.py
```

This opens a browser tab with two modes:

- **Upload**: analyze an image or a recorded video file (same as before).
- **Live check**: opens your webcam in the browser. Click "Take Photo"
  whenever you want a fresh check -- this simulates sampling frames
  periodically during a live call or meeting, the way a real-time liveness
  check would. Each check is scored instantly and added to a running trend
  chart for the session.

**Upgrading live mode to fully automatic streaming:** right now you click to
capture each check. For a no-click, continuously auto-sampling stream (closer
to how a production system like the one in `PRD.md` would run), look into the
`streamlit-webrtc` package, which streams live video into Streamlit and lets
you run your scoring function on a timer against incoming frames. It's a
bigger setup (needs some extra config for camera permissions), so it's listed
here as a natural next step rather than built in from the start.

## Project structure

```
deepfake-detector/
  PRD.md                   # project requirements
  requirements.txt
  README.md
  data/                    # you add this (not included)
  weights/                 # train.py saves model.pt here
  src/
    face_utils.py          # face detection/cropping
    model.py                # model architecture
    dataset.py               # data loading
    train.py                  # training loop
    infer.py                   # scoring an image or video
  scripts/
    prepare_dataset.py         # optional: crop faces from raw photos
  app/
    streamlit_app.py            # the web UI
```

## Known limitations (be upfront about these)

- The face detector (Haar cascade) can miss side-on or partly covered faces.
- A model trained on one dataset usually performs worse on deepfakes made
  with methods it never saw during training. Test with images/videos from a
  source different from your training data to see this for yourself, and
  report it honestly if this is for a course or portfolio.
- This is a probability estimate, not forensic proof.

## Suggested next steps once this works

- Swap the Haar cascade for a stronger face detector (e.g. MTCNN via the
  `facenet-pytorch` package) if you're missing faces often.
- Try a second architecture (e.g. EfficientNet-B0, or a CNN+ViT hybrid as
  described in `PRD.md`) and compare accuracy.
- Evaluate on a dataset you didn't train on (e.g. train on one dataset, test
  on Celeb-DF) to measure real-world generalization.
- Make live mode fully automatic with `streamlit-webrtc` instead of
  click-to-capture.
- Add the voice/speaker-embedding channel and audio-visual (lip-sync)
  consistency check described in the PPT, then fuse both modalities into one
  authenticity score -- this is what would turn the project into the full
  multimodal system it's based on.
