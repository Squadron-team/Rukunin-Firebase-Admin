import re
import numpy as np
import cv2
from typing import List, Tuple, Optional
from .text_segmentation import segment_characters_grayscale
from .data_preprocessing import preprocess_char
from schemas.line_extraction_result import LineExtractionResult
from utils.ml_model_loader import get_model


def predict_char_models(roi: np.ndarray) -> str:
    """
    Run prediction using Logistic regression
    Return a string: (logistic_char)
    """
    processed = preprocess_char(roi)

    if processed is None:
        return ""

    # Logistic regression prediction
    logistic_model = get_model("logistic_regression_ocr")
    log_pred = logistic_model.predict(processed)[0]

    return log_pred


def extract_text_from_line(
    image: np.ndarray, line_boxes: List[Tuple[int, int, int, int]]
) -> str:
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

        # visualize each character
        for i, (cx, cy, cw, ch) in enumerate(char_boxes):
            char_roi = gray_line[cy : cy + ch, cx : cx + cw]

            processed = preprocess_char(char_roi)
            if processed is None:
                print(f"   Character {i}: EMPTY or INVALID")
                continue

            # Predict (optional)
            pred_log = predict_char_models(char_roi)
            text_logistic += pred_log

    return text_logistic


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
) -> List[LineExtractionResult]:
    """Step 3: Extract text from each line using both models"""
    print("\n3. Extracting text from lines")
    lines_data = []

    for i, line_boxes in enumerate(lines):
        text = extract_text_from_line(gray, line_boxes)

        # Skip empty lines
        if not text.strip():
            continue

        # Parse using output
        field, parsed_log = parse_line_by_position(text, i)

        lines_data.append(
            LineExtractionResult(
                line_number=i,
                text=text,
                field=field,
                boxes=line_boxes,
            )
        )

    return lines_data
