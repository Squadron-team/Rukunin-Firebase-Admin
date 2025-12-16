from typing import Dict, List
from schemas.line_extraction_result import LineExtractionResult
from schemas.verification import VerificationResult, FieldVerification, VerificationSummary

def process_verification(
    line_items: List[LineExtractionResult],
    expected_fields: Dict[str, str],
) -> VerificationResult:
    """
    Verifies extracted receipt fields against expected values.
    """
    # ---------------------------------
    # Step 1: Collect detected fields
    # ---------------------------------
    detected_fields: Dict[str, str] = {}

    for item in line_items:
        if not item.field:
            continue

        # Keep first occurrence only
        if item.field not in detected_fields:
            detected_fields[item.field] = item.text

    # ---------------------------------
    # Step 2: Verify against expectations
    # ---------------------------------
    field_results: Dict[str, FieldVerification] = {}

    for f, expected_value in expected_fields.items():
        actual_value = detected_fields.get(f)

        if actual_value is None:
            continue

        field_results[f] = FieldVerification(
            expected=expected_value,
            actual=actual_value,
            is_match=expected_value.strip() == actual_value.strip(),
        )

    # ---------------------------------
    # Step 3: Final verdict
    # ---------------------------------
    passed = True
    for result in field_results.values():
        if not result.is_match:
            passed = False
            break

    summary = VerificationSummary(
        passed=passed,
        final_verdict="VALID RECEIPT" if passed else "POTENTIALLY FAKE",
    )

    return VerificationResult(
        field_verification=field_results,
        summary=summary,
    )