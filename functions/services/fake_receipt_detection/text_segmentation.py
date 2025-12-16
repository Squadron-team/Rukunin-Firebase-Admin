import numpy as np
import cv2
from typing import Tuple, List
from cv2.typing import Rect


def process_text_detection(
    image: np.ndarray,
) -> Tuple[
    np.ndarray, List[Tuple[int, int, int, int]], np.ndarray, np.ndarray, np.ndarray
]:
    """Your existing function - this works well"""
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 20 or h < 10:
            continue
        ratio = w / float(h)
        if ratio < 0.2 or ratio > 15:
            continue
        area = w * h
        if area < 50 or area > image.shape[0] * image.shape[1] * 0.5:
            continue

        boxes.append((x, y, w, h))
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return original, boxes, gray, edges, dilated


def process_line_grouping(
    boxes: List[Tuple[int, int, int, int]], line_threshold: int = 15
) -> List[List[Tuple[int, int, int, int]]]:
    """Group boxes that are on the same line"""
    print("\n2.1 Line Grouping")

    if not boxes:
        return []

    # Sort by Y position
    boxes_sorted = sorted(boxes, key=lambda box: box[1])

    lines = []
    current_line = []
    current_y = boxes_sorted[0][1]

    for box in boxes_sorted:
        x, y, w, h = box

        # Check if this box is on the same line (within threshold)
        if abs(y - current_y) <= line_threshold:
            current_line.append(box)
        else:
            # New line detected
            if current_line:
                # Sort boxes in the line by X coordinate (left to right)
                current_line_sorted = sorted(current_line, key=lambda b: b[0])
                lines.append(current_line_sorted)
            current_line = [box]
            current_y = y

    # Don't forget the last line
    if current_line:
        current_line_sorted = sorted(current_line, key=lambda b: b[0])
        lines.append(current_line_sorted)

    return lines


def merge_small_vertical_components(
    boxes: List[Rect],
    y_thresh: int = 10,
    x_overlap_ratio: float = 0.5,
    handle_colons: bool = True,
    colon_y_thresh: int = 35,
    colon_h_align_ratio: float = 0.6,
) -> List[Tuple[int, int, int, int]]:
    """
    Merge vertically separated components of the same character (like 'i', 'j', ':').

    Args:
        boxes: List of bounding boxes (x, y, w, h)
        y_thresh: Maximum vertical distance for general merging (i, j)
        x_overlap_ratio: Minimum horizontal overlap ratio required
        handle_colons: Whether to apply special logic for colons
        colon_y_thresh: Larger threshold for colon detection (vertical gap)
        colon_h_align_ratio: Horizontal alignment tolerance for colons

    Returns:
        List of merged bounding boxes
    """
    merged = []
    used = set()

    for i, (x1, y1, w1, h1) in enumerate(boxes):
        if i in used:
            continue

        merged_box = (x1, y1, w1, h1)
        merged_components = [i]

        # Keep searching for components to merge
        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i == j or j in used:
                continue

            # Update coordinates from current merged box
            mx, my, mw, mh = merged_box

            # Calculate vertical gap between boxes
            vertical_gap = min(abs((my + mh) - y2), abs((y2 + h2) - my))

            # Calculate horizontal overlap
            x_overlap = max(0, min(mx + mw, x2 + w2) - max(mx, x2))
            has_horizontal_overlap = x_overlap > min(mw, w2) * x_overlap_ratio

            # Check horizontal alignment (for colons)
            center_x1 = mx + mw / 2
            center_x2 = x2 + w2 / 2
            h_alignment = abs(center_x1 - center_x2) / max(mw, w2)
            is_horizontally_aligned = h_alignment < colon_h_align_ratio

            should_merge = False

            # Standard close merging (for 'i', 'j')
            if vertical_gap < y_thresh and has_horizontal_overlap:
                should_merge = True

            # Special colon handling
            if (
                handle_colons
                and vertical_gap < colon_y_thresh
                and is_horizontally_aligned
            ):
                # Check if components are small and similar in size
                area1 = mw * mh
                area2 = w2 * h2
                avg_area = (area1 + area2) / 2
                size_ratio = (
                    min(area1, area2) / max(area1, area2)
                    if max(area1, area2) > 0
                    else 0
                )

                # Both are small dots and similar size → likely colon
                if avg_area < 300 and size_ratio > 0.25:
                    should_merge = True

            if should_merge:
                # Merge into expanded bounding box
                nx = min(mx, x2)
                ny = min(my, y2)
                nw = max(mx + mw, x2 + w2) - nx
                nh = max(my + mh, y2 + h2) - ny

                merged_box = (nx, ny, nw, nh)
                merged_components.append(j)
                used.add(j)

        merged.append(merged_box)
        for comp_idx in merged_components:
            used.add(comp_idx)

    return merged


def segment_characters_grayscale(
    image_gray: np.ndarray,
    merge_vertical: bool = True,
    y_thresh: int = 10,
    x_overlap_ratio: float = 0.5,
) -> List[Rect]:
    """
    Segment characters from grayscale image with optional vertical merging.

    Args:
        image_gray: Grayscale input image
        merge_vertical: Whether to merge vertically separated components
        y_thresh: Vertical distance threshold for merging
        x_overlap_ratio: Horizontal overlap ratio for merging

    Returns:
        List of character bounding boxes sorted left to right
    """
    _, bw = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]

    # Merge vertical components for characters like 'i', 'j'
    if merge_vertical and len(boxes) > 0:
        boxes = merge_small_vertical_components(boxes, y_thresh, x_overlap_ratio)

    # Sort left to right
    boxes = sorted(boxes, key=lambda b: b[0])
    return boxes


def segment_characters(
    img_bgr: np.ndarray,
    merge_vertical: bool = True,
    y_thresh: int = 10,
    x_overlap_ratio: float = 0.5,
) -> List[Rect]:
    """
    Segment characters from BGR image with optional vertical merging.

    Args:
        img_bgr: BGR input image
        merge_vertical: Whether to merge vertically separated components
        y_thresh: Vertical distance threshold for merging
        x_overlap_ratio: Horizontal overlap ratio for merging

    Returns:
        List of character bounding boxes sorted left to right
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find contours (characters)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # bounding boxes
    boxes = [cv2.boundingRect(c) for c in contours]

    # Merge vertical components for characters like 'i', 'j'
    if merge_vertical and len(boxes) > 0:
        boxes = merge_small_vertical_components(boxes, y_thresh, x_overlap_ratio)

    # sort left to right
    boxes = sorted(boxes, key=lambda b: b[0])

    return boxes
