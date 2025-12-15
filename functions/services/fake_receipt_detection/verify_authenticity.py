from typing import List, Dict, Any

def process_verification(
    line_items: List[Dict[str, Any]], expected_fields: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verifies extracted receipt fields using both OCR classifiers:
    - KNN
    - Logistic Regression

    Parameters:
        line_items: list of dicts (each contains OCR results)
        expected_fields: dict defining expected values or formats

    Returns:
        dict summary of:
        - knn_verification
        - logistic_verification
        - combined_verdict
    """

    results_knn = {}
    results_logistic = {}

    # --------------------------
    # PROCESS EACH EXTRACTED FIELD
    # --------------------------
    for item in line_items:
        # Each item contains OCR outputs for both models
        field_k = item.get("field_knn")
        text_k = item.get("text_knn")

        field_l = item.get("field_logistic")
        text_l = item.get("text_logistic")

        # --------------------------
        # Store KNN result
        # --------------------------
        if field_k:
            if field_k not in results_knn:
                results_knn[field_k] = text_k

        # --------------------------
        # Store Logistic result
        # --------------------------
        if field_l:
            if field_l not in results_logistic:
                results_logistic[field_l] = text_l

    # ---------------------------------
    # VERIFICATION LOGIC
    # ---------------------------------
    knn_check = {}
    logistic_check = {}

    def check_value(expected, actual):
        """Simple comparison helper."""
        if expected is None:
            return True
        return str(expected).strip() == str(actual).strip()

    # --------------------------
    # Check KNN
    # --------------------------
    for field, expected_value in expected_fields.items():
        print("KNN verification")
        actual = results_knn.get(field)
        print(f"Actual {actual} - Expected value: {expected_value}")
        knn_check[field] = {
            "expected": expected_value,
            "actual": actual,
            "match": check_value(expected_value, actual),
        }

    # --------------------------
    # Check Logistic Regression
    # --------------------------
    for field, expected_value in expected_fields.items():
        print("Logistic regression verification")
        actual = results_logistic.get(field)
        print(f"Actual {actual} - Expected value: {expected_value}")
        logistic_check[field] = {
            "expected": expected_value,
            "actual": actual,
            "match": check_value(expected_value, actual),
        }

    # --------------------------
    # Combined final verdict
    # --------------------------
    knn_pass = all(v["match"] for v in knn_check.values())
    logistic_pass = all(v["match"] for v in logistic_check.values())

    combined = {
        "knn_pass": knn_pass,
        "logistic_pass": logistic_pass,
        "final_verdict": (
            "VALID RECEIPT" if knn_pass or logistic_pass else "POTENTIALLY FAKE"
        ),
    }

    # --------------------------
    # Return full report
    # --------------------------
    return {
        "knn_verification": knn_check,
        "logistic_verification": logistic_check,
        "combined": combined,
    }