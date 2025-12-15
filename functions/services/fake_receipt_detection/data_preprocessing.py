import numpy as np
import cv2
from typing import Optional

def preprocess_char(roi: np.ndarray) -> Optional[np.ndarray]:
    """
    Preprocess cropped ROI for model inference using
    EXACT SAME STEPS as training preprocess_digit().
    """

    # 1. ROI is already grayscale, no need for BGR2GRAY
    gray = roi.copy()

    # 2. threshold (binary inverse + Otsu)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. find bounding box (same as training)
    ys, xs = np.where(bw == 255)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    digit = bw[y1 : y2 + 1, x1 : x2 + 1]

    # 4. resize longest side to 22 px
    h, w = digit.shape
    if h > w:
        new_h, new_w = 22, int(w * (22 / h))
    else:
        new_w, new_h = 22, int(h * (22 / w))

    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5. pad to 28x28
    top = (28 - new_h) // 2
    bottom = 28 - new_h - top
    left = (28 - new_w) // 2
    right = 28 - new_w - left

    digit_padded = cv2.copyMakeBorder(
        digit_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )

    # 6. normalize + flatten
    digit_norm = digit_padded.astype("float32") / 255.0
    flat = digit_norm.flatten()

    return flat.reshape(1, -1)