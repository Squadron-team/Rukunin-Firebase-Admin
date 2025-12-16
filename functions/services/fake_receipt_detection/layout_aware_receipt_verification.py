import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any
from .text_segmentation import process_line_grouping, process_text_detection
from .text_extraction import process_text_extraction
from .verify_authenticity import process_verification
from schemas.verification import VerificationResult, VerificationSummary
from schemas.line_count_validation import LineCountValidationResult
from schemas.line_extraction_result import LineExtractionResult
from schemas.layout_validation_result import LayoutValidationResult
from schemas.layout_fingerprint import LayoutFingerprint

def get_field_colors() -> Dict[str, Tuple[int, int, int]]:
    """Get color mapping for different field types"""
    return {
        "app_logo": (255, 0, 0),  # Red
        "bank_name": (0, 0, 255),  # Blue
        "transaction_status": (0, 255, 0),  # Green
        "total_amount": (255, 255, 0),  # Cyan
        "datetime": (255, 0, 255),  # Magenta
        "transaction_id": (0, 255, 255),  # Yellow
        "receiver_title": (128, 0, 128),  # Purple
        "receiver_name": (128, 128, 0),  # Olive
        "receiver_bank": (0, 128, 128),  # Teal
        "sender_title": (128, 128, 128),  # Gray
        "sender_name": (64, 64, 64),  # Dark Gray
        "sender_account": (192, 192, 192),  # Light Gray
        "detail_title": (255, 165, 0),  # Orange
        "nominal_line": (255, 192, 203),  # Pink
        "fee_line": (173, 216, 230),  # Light Blue
        "total_line": (144, 238, 144),  # Light Green
        "unknown": (0, 0, 0),  # Black
    }


def extract_layout_fingerprint(
    lines: List[List[Tuple[int, int, int, int]]], image_shape: Tuple[int, int]
) -> LayoutFingerprint:
    """
    Extract structural features from receipt layout to create a fingerprint.

    Args:
        lines: List of line segments (each containing boxes)
        image_shape: (height, width) of the image

    Returns:
        Dictionary containing layout features
    """
    if not lines:
        return LayoutFingerprint()

    img_height, img_width = image_shape

    line_y_positions: List[float] = []
    line_heights: List[float] = []
    line_widths: List[float] = []
    line_x_starts: List[float] = []

    for line_boxes in lines:
        if not line_boxes:
            continue

        y_pos = line_boxes[0][1]
        line_y_positions.append(y_pos / img_height)

        y_min = min(box[1] for box in line_boxes)
        y_max = max(box[1] + box[3] for box in line_boxes)
        x_min = min(box[0] for box in line_boxes)
        x_max = max(box[0] + box[2] for box in line_boxes)

        line_heights.append((y_max - y_min) / img_height)
        line_widths.append((x_max - x_min) / img_width)
        line_x_starts.append(x_min / img_width)

    line_gaps: List[float] = [
        line_y_positions[i + 1] - line_y_positions[i]
        for i in range(len(line_y_positions) - 1)
    ]

    alignment_variance = float(np.var(line_x_starts)) if line_x_starts else 0.0

    total_lines = len(line_y_positions)

    zone_distribution = (
        [
            sum(1 for y in line_y_positions if y < 0.33) / total_lines,
            sum(1 for y in line_y_positions if 0.33 <= y < 0.67) / total_lines,
            sum(1 for y in line_y_positions if y >= 0.67) / total_lines,
        ]
        if total_lines
        else [0.0, 0.0, 0.0]
    )

    width_distribution = (
        [
            sum(1 for w in line_widths if w < 0.5) / total_lines,
            sum(1 for w in line_widths if 0.5 <= w < 0.8) / total_lines,
            sum(1 for w in line_widths if w >= 0.8) / total_lines,
        ]
        if total_lines
        else [0.0, 0.0, 0.0]
    )

    return LayoutFingerprint(
        num_lines=len(lines),
        avg_line_height=float(np.mean(line_heights)) if line_heights else 0.0,
        std_line_height=float(np.std(line_heights)) if line_heights else 0.0,
        avg_line_width=float(np.mean(line_widths)) if line_widths else 0.0,
        std_line_width=float(np.std(line_widths)) if line_widths else 0.0,
        avg_line_gap=float(np.mean(line_gaps)) if line_gaps else 0.0,
        std_line_gap=float(np.std(line_gaps)) if line_gaps else 0.0,
        alignment_variance=alignment_variance,
        zone_distribution=zone_distribution,
        width_distribution=width_distribution,
        line_y_positions=line_y_positions,
    )


def get_template_layout_fingerprint() -> Dict[str, Any]:
    """
    Define the expected layout fingerprint for valid receipts.
    These values should be calibrated based on your genuine receipt samples.

    Returns:
        Dictionary containing expected layout features with acceptable ranges
    """
    return {
        "num_lines": {"min": 10, "max": 20, "ideal": 15},
        "avg_line_height": {"min": 0.02, "max": 0.08, "ideal": 0.04},
        "avg_line_width": {"min": 0.5, "max": 0.95, "ideal": 0.75},
        "avg_line_gap": {"min": 0.02, "max": 0.10, "ideal": 0.05},
        "alignment_variance": {"max": 0.05},  # Lines should be mostly aligned
        # Zone distribution: [top%, middle%, bottom%]
        # Expected: more content in top and middle sections
        "zone_distribution": {
            "top": {"min": 0.2, "max": 0.5},
            "middle": {"min": 0.3, "max": 0.6},
            "bottom": {"min": 0.1, "max": 0.4},
        },
        # Width distribution: [narrow%, medium%, wide%]
        "width_distribution": {
            "narrow": {"max": 0.3},  # Few narrow lines
            "medium": {"min": 0.2, "max": 0.6},
            "wide": {"min": 0.3, "max": 0.8},  # Many wide lines
        },
    }


def calculate_layout_similarity(
    detected: LayoutFingerprint, template: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate similarity score between detected layout and template.

    Args:
        detected: Extracted layout fingerprint from image
        template: Expected layout fingerprint

    Returns:
        Dictionary containing similarity scores and detailed analysis
    """
    scores = {}
    violations = []

    # 1. Number of lines check
    num_lines = detected.num_lines
    if template["num_lines"]["min"] <= num_lines <= template["num_lines"]["max"]:
        scores["num_lines"] = 1.0
    else:
        scores["num_lines"] = 0.0
        violations.append(
            f"Line count {num_lines} outside range [{template['num_lines']['min']}, {template['num_lines']['max']}]"
        )

    # 2. Average line height check
    avg_height = detected.avg_line_height
    if (
        template["avg_line_height"]["min"]
        <= avg_height
        <= template["avg_line_height"]["max"]
    ):
        scores["avg_line_height"] = 1.0
    else:
        scores["avg_line_height"] = 0.0
        violations.append(
            f"Average line height {avg_height:.3f} outside expected range"
        )

    # 3. Average line width check
    avg_width = detected.avg_line_width
    if (
        template["avg_line_width"]["min"]
        <= avg_width
        <= template["avg_line_width"]["max"]
    ):
        scores["avg_line_width"] = 1.0
    else:
        scores["avg_line_width"] = 0.0
        violations.append(f"Average line width {avg_width:.3f} outside expected range")

    # 4. Line spacing consistency
    avg_gap = detected.avg_line_gap
    if template["avg_line_gap"]["min"] <= avg_gap <= template["avg_line_gap"]["max"]:
        scores["avg_line_gap"] = 1.0
    else:
        scores["avg_line_gap"] = 0.0
        violations.append(f"Line spacing {avg_gap:.3f} inconsistent with template")

    # 5. Alignment check
    alignment_var = detected.alignment_variance
    if alignment_var <= template["alignment_variance"]["max"]:
        scores["alignment"] = 1.0
    else:
        scores["alignment"] = 0.0
        violations.append(f"Poor alignment detected (variance: {alignment_var:.3f})")

    # 6. Zone distribution check
    zone_dist = detected.zone_distribution
    zone_score = 0
    zone_checks = 0

    zones = ["top", "middle", "bottom"]
    for i, zone in enumerate(zones):
        zone_checks += 1
        zone_val = zone_dist[i]
        zone_template = template["zone_distribution"][zone]

        if zone_template["min"] <= zone_val <= zone_template["max"]:
            zone_score += 1
        else:
            violations.append(
                f"{zone.capitalize()} zone density {zone_val:.2f} outside [{zone_template['min']}, {zone_template['max']}]"
            )

    scores["zone_distribution"] = zone_score / zone_checks if zone_checks > 0 else 0

    # 7. Width distribution check
    width_dist = detected.width_distribution
    width_score = 0
    width_checks = 0

    width_types = ["narrow", "medium", "wide"]
    for i, width_type in enumerate(width_types):
        width_checks += 1
        width_val = width_dist[i]
        width_template = template["width_distribution"][width_type]

        if "min" in width_template and "max" in width_template:
            if width_template["min"] <= width_val <= width_template["max"]:
                width_score += 1
            else:
                violations.append(
                    f"{width_type.capitalize()} lines ratio {width_val:.2f} outside expected range"
                )
        elif "max" in width_template:
            if width_val <= width_template["max"]:
                width_score += 1
            else:
                violations.append(f"Too many {width_type} lines: {width_val:.2f}")

    scores["width_distribution"] = width_score / width_checks if width_checks > 0 else 0

    # Calculate overall similarity score (weighted average)
    weights = {
        "num_lines": 0.20,
        "avg_line_height": 0.10,
        "avg_line_width": 0.10,
        "avg_line_gap": 0.10,
        "alignment": 0.15,
        "zone_distribution": 0.20,
        "width_distribution": 0.15,
    }

    overall_score = sum(scores[key] * weights[key] for key in scores.keys())

    return {
        "overall_score": overall_score,
        "detailed_scores": scores,
        "violations": violations,
        "is_similar": overall_score >= 0.65,  # Threshold: 65% similarity required
    }


def validate_layout_structure(
    lines: List[List[Tuple[int, int, int, int]]],
    image_shape: Tuple[int, int],
    similarity_threshold: float = 0.65,
) -> LayoutValidationResult:
    """
    Validate if the detected layout structure matches the expected receipt template.
    """
    print("2.3 Validate layout structure")

    detected_fingerprint = extract_layout_fingerprint(lines, image_shape)
    template_fingerprint = get_template_layout_fingerprint()

    similarity_result = calculate_layout_similarity(
        detected_fingerprint, template_fingerprint
    )

    is_valid = similarity_result["is_similar"]
    overall_score = similarity_result["overall_score"]

    reason = ""

    if not is_valid:
        reason = (
            f"Layout structure mismatch (similarity: {overall_score:.1%}). "
            "This receipt has a different layout than expected. "
            "Likely FAKE or wrong receipt type."
        )

        if similarity_result["violations"]:
            reason += (
                f"\nKey violations: {'; '.join(similarity_result['violations'][:3])}"
            )
    else:
        reason = (
            f"Layout structure matches expected template "
            f"(similarity: {overall_score:.1%})."
        )

    return LayoutValidationResult(
        is_valid=is_valid,
        similarity_score=overall_score,
        threshold=similarity_threshold,
        detected_fingerprint=detected_fingerprint,
        template_fingerprint=template_fingerprint,
        detailed_scores=similarity_result["detailed_scores"],
        violations=similarity_result["violations"],
        reason=reason,
    )


def validate_line_count(
    lines: List[List[Tuple[int, int, int, int]]],
    correct_lines: int,
) -> LineCountValidationResult:
    """
    Validate if the number of detected line segments is within acceptable range.
    Receipts with too many or too few lines are likely fake/edited.

    Args:
        lines: List of line segments (each containing boxes)
        min_lines: Minimum expected line count for a valid receipt
        max_lines: Maximum expected line count for a valid receipt

    Returns:
        Dictionary containing validation result and details
    """
    print("2.2 Validate line count")
    
    num_lines = len(lines)
    is_valid = correct_lines == num_lines

    if num_lines != correct_lines:
        reason = f"Detected ({num_lines}). Total detected lines not match with the correct one. Likely FAKE or edited image."
    else:
        reason = f"Line count ({num_lines}) is within acceptable range."

    return LineCountValidationResult(
        is_valid=is_valid,
        detected_lines=num_lines,
        expected_lines=correct_lines,
        reason=reason,
    )


def display_verification_results(verification: VerificationResult) -> None:
    """Step 5: Display verification results in console"""
    print("\n5. VERIFICATION RESULT")

    # Display layout structure validation if present
    if verification.layout_validation is not None:
        layout_val = verification.layout_validation
        print("\n--- Layout Structure Validation ---")
        print(f"Similarity Score: {layout_val.similarity_score:.1%}")
        print(f"Threshold       : {layout_val.threshold:.1%}")
        print(f"Status          : {'PASS' if layout_val.is_valid else 'FAIL'}")
        print(f"Reason          : {layout_val.reason}")

        # Show detailed scores
        if layout_val.detailed_scores:
            print("\nDetailed Scores:")
            for metric, score in layout_val.detailed_scores.items():
                status = "✓" if score >= 0.5 else "✗"
                print(f"  {status} {metric}: {score:.1%}")

        if not layout_val.is_valid:
            print("\n EARLY DETECTION: Receipt is FAKE based on layout structure!")
            return

    print(f"FINAL VERDICT   : {verification.summary.final_verdict}")


def create_legend_image(lines_data: List[LineExtractionResult]) -> np.ndarray:
    """Create legend image showing detected field types"""
    colors = get_field_colors()
    legend_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    y_offset = 30

    for field_type, color in colors.items():
        if any(field_type == line.field for line in lines_data):
            cv2.putText(
                legend_img,
                field_type,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )
            y_offset += 20

    return legend_img


def visualize_results(
    image: np.ndarray,
    result_img: np.ndarray,
    annotated_img: np.ndarray,
    lines_data: List[LineExtractionResult],
) -> None:
    """Step 6: Create and display visualization"""
    print("\n6. Create visualization")
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("Text Detection")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    plt.title("Layout Analysis")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    legend_img = create_legend_image(lines_data)
    plt.imshow(cv2.cvtColor(legend_img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Field Types")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def create_annotated_image(
    image: np.ndarray,
    lines_data: List[LineExtractionResult],
    verification: VerificationResult,
) -> np.ndarray:
    """Create annotated image with bounding boxes and labels"""
    annotated_img = image.copy()
    colors = get_field_colors()

    # Draw annotated boxes
    for line_data in lines_data:
        color = colors.get(line_data.field, (0, 0, 0))

        for box in line_data.boxes:
            x, y, w, h = box
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)

        # Add field type label using first box in line
        if line_data.boxes:
            x, y, w, h = line_data.boxes[0]
            label = line_data.field
            cv2.putText(
                annotated_img,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    # Add final result to image
    final_result = verification.summary.final_verdict

    result_text = f"Result: {final_result}"

    cv2.putText(
        annotated_img, result_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3
    )

    return annotated_img

def layout_aware_receipt_verification(
    image: np.ndarray, expected_amount: Dict[str, Any]
) -> Tuple[VerificationResult, List[LineExtractionResult]]:
    """Complete pipeline with layout awareness"""
    print("=== LAYOUT-AWARE RECEIPT VERIFICATION ===")

    # Step 1: Detect text regions
    result_img, boxes, gray, edges, dilated = process_text_detection(image)

    # Step 2.1: Group boxes by lines
    lines = process_line_grouping(boxes, line_threshold=15)

    # Step 2.2: EARLY VALIDATION - Check line count
    line_validation = validate_line_count(lines, correct_lines=15)

    if not line_validation.is_valid:
        # Create early fake detection result
        verification = VerificationResult(
            summary=VerificationSummary(
                passed=False,
                final_verdict="POTENTIALLY FAKE",
            ),
            line_validation=line_validation,
        )

        # Display results
        display_verification_results(verification)

        # Create simple annotated image showing fake detection
        annotated_img = image.copy()
        cv2.putText(
            annotated_img,
            "FAKE RECEIPT DETECTED!",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),  # Red color
            3,
        )
        cv2.putText(
            annotated_img,
            line_validation.reason,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        visualize_results(image, result_img, annotated_img, [])

        return verification, []

    # Step 2.3: ADVANCED VALIDATION - Check layout structure
    layout_validation = validate_layout_structure(
        lines, image_shape=(image.shape[0], image.shape[1]), similarity_threshold=0.65
    )

    if not layout_validation.is_valid:
        # Create early fake detection result
        verification = VerificationResult(
            summary=VerificationSummary(passed=False, final_verdict="POTENTIALLY FAKE"),
            line_validation=line_validation,
            layout_validation=layout_validation,
        )

        # Display results
        display_verification_results(verification)

        # Create annotated image showing layout mismatch
        annotated_img = image.copy()
        cv2.putText(
            annotated_img,
            "FAKE RECEIPT DETECTED!",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )
        cv2.putText(
            annotated_img,
            f"Layout Similarity: {layout_validation.similarity_score:.1%}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            annotated_img,
            "Different layout structure detected!",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        visualize_results(image, result_img, annotated_img, [])

        return verification, []

    # Step 3: Extract text from each line
    lines_data = process_text_extraction(gray, lines)

    # Step 4: Verify authenticity
    verification = process_verification(lines_data, expected_amount)

    # Add line validation to verification results
    verification.line_validation = line_validation
    verification.layout_validation = layout_validation
    verification.early_detection = False

    # Step 5: Display results
    display_verification_results(verification)

    # Step 6: Create visualization
    annotated_img = create_annotated_image(image, lines_data, verification)
    visualize_results(image, result_img, annotated_img, lines_data)

    return verification, lines_data