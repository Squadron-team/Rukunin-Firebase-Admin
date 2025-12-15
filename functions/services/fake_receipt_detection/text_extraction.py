import re
import numpy as np
import cv2
from typing import Tuple, List, Optional, Dict, Any
from .text_segmentation import segment_characters_grayscale
from .data_preprocessing import preprocess_char
from utils.ml_model_loader import get_model

def predict_char_models(roi: np.ndarray) -> Tuple[str, str]:
    """
    Run prediction using both models.
    Return a tuple: (knn_char, logistic_char)
    """
    processed = preprocess_char(roi)

    if processed is None:
        return "", ""

    # KNN prediction
    knn_model = get_model("knn_ocr")
    knn_pred = knn_model.predict(processed)[0]

    # Logistic regression prediction
    logistic_model = get_model("logistic_regression_ocr")
    log_pred = logistic_model.predict(processed)[0]

    return knn_pred, log_pred


def extract_text_from_line(
    image: np.ndarray, line_boxes: List[Tuple[int, int, int, int]]
) -> Tuple[str, str]:

    text_knn = ""
    text_logistic = ""

    for x, y, w, h in line_boxes:
        # crop line region
        line_roi = image[y : y + h, x : x + w]

        # convert to grayscale if needed
        if len(line_roi.shape) == 3:
            gray_line = cv2.cvtColor(line_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_line = line_roi

        # segment characters inside this line - use grayscale version
        char_boxes = segment_characters_grayscale(gray_line)
        
        # Extract text from each character
        for cx, cy, cw, ch in char_boxes:
            char_roi = gray_line[cy : cy + ch, cx : cx + cw]
            
            # Predict character using both models
            knn_char, log_char = predict_char_models(char_roi)
            
            text_knn += knn_char
            text_logistic += log_char
        
        # Add space between boxes in the same line
        text_knn += " "
        text_logistic += " "
        
    return text_knn.strip(), text_logistic.strip()


def parse_line_by_position(line_text: str, line_number: int) -> Tuple[str, str]:
    t = line_text.lower()

    # Strong keyword-based classification
    if any(k in t for k in ["transfer berhasil", "berhasil"]):
        return "transaction_status", line_text

    if re.search(r"rp\s*\d", t):
        return "total_amount", line_text

    if any(
        m in t
        for m in [
            "wib",
            "jan",
            "feb",
            "mar",
            "apr",
            "mei",
            "jun",
            "jul",
            "agu",
            "sep",
            "okt",
            "nov",
            "des",
        ]
    ):
        return "datetime", line_text

    if "ref" in t and "id" in t:
        return "transaction_id", line_text

    if "penerima" in t:
        return "receiver_title", line_text

    # Name heuristics
    if any(prefix in t for prefix in ["sdri", "sdr", "bpk", "ibu"]):
        return "receiver_name", line_text

    if any(b in t for b in ["bni", "bca", "mandiri", "bri", "btn"]):
        return "receiver_bank", line_text

    if "sumber dana" in t:
        return "sender_title", line_text

    # Masked account
    if "***" in line_text or re.search(r"[xX]{4,}", t):
        return "sender_account", line_text

    # Detail transfer
    if "detail transfer" in t:
        return "detail_title", line_text

    # Nominal, Fee, Total lines
    if "nominal" in t:
        return "nominal_line", line_text
    if "biaya" in t:
        return "fee_line", line_text
    if t.startswith("total") or (" total " in t):
        return "total_line", line_text

    return "unknown", line_text


def extract_amount_from_text(text: str) -> Optional[int]:
    """Extract amount value from text"""
    amount_match = re.search(r"Rp\s*([\d.,]+)", text.replace(".", ""))
    if amount_match:
        amount_str = amount_match.group(1).replace(",", "")
        return int(amount_str) if amount_str.isdigit() else None
    return None


def parse_detail_line(text: str) -> Tuple[str, Optional[int]]:
    """Parse detail lines that have labels and amounts separated"""
    if "nominal" in text.lower():
        return "nominal", extract_amount_from_text(text)
    elif "biaya transaksi" in text.lower():
        return "fee", extract_amount_from_text(text)
    elif "total" in text.lower():
        return "total", extract_amount_from_text(text)
    return "unknown", None

def process_text_extraction(
    gray: np.ndarray, lines: List[List[Tuple[int, int, int, int]]]
) -> List[Dict[str, Any]]:
    """Step 3: Extract text from each line using both models"""
    lines_data = []

    for i, line_boxes in enumerate(lines):
        text_knn, text_logistic = extract_text_from_line(gray, line_boxes)

        # Skip empty lines
        if (not text_knn.strip()) and (not text_logistic.strip()):
            continue

        # Parse using BOTH outputs
        field_knn, parsed_knn = parse_line_by_position(text_knn, i)
        field_log, parsed_log = parse_line_by_position(text_logistic, i)

        lines_data.append(
            {
                "line_number": i,
                "text_knn": text_knn,
                "text_logistic": text_logistic,
                "field_knn": field_knn,
                "field_logistic": field_log,
                "boxes": line_boxes,
            }
        )

    return lines_data