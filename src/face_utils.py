"""
Face detection and cropping helpers.

Uses OpenCV's built-in Haar cascade face detector, so no extra model download
is needed. It is fast but not the most robust (it can miss side-on faces or
faces partly covered by something). If you want better detection later, swap
`find_faces` to use MTCNN (from the `facenet-pytorch` package) or MediaPipe
Face Detection -- everything else in this project only needs a list of boxes.
"""

import os

import cv2
import numpy as np

# As of OpenCV 5.0, CascadeClassifier (and its bundled XML data files) moved
# into a contrib module, so cv2.data.haarcascades is no longer reliable
# across installs. To avoid depending on it, this project ships its own copy
# of the cascade file right next to this script -- see the README for the
# one-time download command if it's missing.
_LOCAL_CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml"
)


class FaceCropper:
    def __init__(self, margin: float = 0.3, min_face_size: int = 40):
        """
        margin: extra padding around the detected face box, as a fraction of
            the face size, so the crop includes some context (hairline, chin).
        min_face_size: ignore detections smaller than this many pixels wide,
            to filter out false positives on noise/texture.
        """
        if os.path.exists(_LOCAL_CASCADE_PATH):
            cascade_path = _LOCAL_CASCADE_PATH
        else:
            # Fall back to OpenCV's bundled copy, for older OpenCV versions
            # where this still works.
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(
                f"Could not load face cascade from {cascade_path}. "
                "Run the download command in the README's Troubleshooting "
                "section to fetch haarcascade_frontalface_default.xml into "
                "the src/ folder."
            )
        self.margin = margin
        self.min_face_size = min_face_size

    def find_faces(self, rgb_image: np.ndarray):
        """
        rgb_image: HxWx3 uint8 array in RGB order.
        Returns a list of (x, y, w, h) boxes, one per detected face.
        """
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)  # improves detection in uneven lighting
        boxes = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        return [tuple(int(v) for v in box) for box in boxes]

    def crop_largest_face(self, rgb_image: np.ndarray):
        """
        Returns a square RGB crop around the largest detected face,
        or None if no face was found.
        """
        boxes = self.find_faces(rgb_image)
        if not boxes:
            return None

        # Pick the face with the largest area -- usually the main subject.
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])

        # Expand the box by `margin` on each side, keeping it square.
        side = int(max(w, h) * (1 + 2 * self.margin))
        cx, cy = x + w // 2, y + h // 2
        x0, y0 = cx - side // 2, cy - side // 2
        x1, y1 = x0 + side, y0 + side

        # If the box runs past the image edges, pad the image instead of
        # cropping a non-square region.
        height, width = rgb_image.shape[:2]
        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - width)
        pad_bottom = max(0, y1 - height)

        if pad_left or pad_top or pad_right or pad_bottom:
            rgb_image = cv2.copyMakeBorder(
                rgb_image, pad_top, pad_bottom, pad_left, pad_right,
                borderType=cv2.BORDER_REPLICATE,
            )
            x0, x1 = x0 + pad_left, x1 + pad_left
            y0, y1 = y0 + pad_top, y1 + pad_top

        return rgb_image[y0:y1, x0:x1]