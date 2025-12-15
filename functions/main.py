# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app
import base64
import logging
from PIL import Image
from io import BytesIO
import numpy as np
import sys
from services.image_processing import ImagePreprocessor, HOGExtractor
from utils.ml_model_loader import get_model
from services.fake_receipt_detection.text_segmentation import process_text_detection, process_line_grouping
from services.fake_receipt_detection.text_extraction import process_text_extraction
from services.fake_receipt_detection.verify_authenticity import process_verification
from services.fake_receipt_detection.layout_aware_receipt_verification import (
    validate_layout_structure,
    validate_line_count
)
import cv2

# Fix for loading pickled scikit-learn pipelines:
# The model was trained in an environment where custom transformers
# (ImagePreprocessor, HOGExtractor) were defined under the __main__ module.
# Pickle stores class locations, so we alias these classes into __main__
# to ensure joblib.load() can reconstruct the pipeline correctly.
setattr(sys.modules["__main__"], "ImagePreprocessor", ImagePreprocessor)
setattr(sys.modules["__main__"], "HOGExtractor", HOGExtractor)

# For cost control, you can set the maximum number of containers that can be
# running at the same time. This helps mitigate the impact of unexpected
# traffic spikes by instead downgrading performance. This limit is a per-functionfir
# limit. You can override the limit for each function using the max_instances
# parameter in the decorator, e.g. @https_fn.on_request(max_instances=5).
set_global_options(max_instances=2)

initialize_app()

@https_fn.on_call()
def classify_image(req: https_fn.CallableRequest):
    """
    HTTP Cloud Function for image classification
    Expects JSON with 'image' field containing base64 encoded image
    Returns JSON with prediction results
    """
    data = req.data

    img_base64 = data.get("image")
    if not img_base64:
        return https_fn.HttpsError(
            https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            "Missing 'image' field"
        )

    try:
        # Decode base64 into bytes
        img_bytes = base64.b64decode(img_base64)

        # Open as PIL image
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        # Load model (cached)
        model = get_model("simple_ml_classifier")

        if model is None:
            raise Exception("Model is not loaded properly!")

        prediction = model.predict([img_np])[0]  # type: ignore
        probabilities = model.predict_proba([img_np])[0].tolist()  # type: ignore

        return {
            "success": True,
            "prediction": int(prediction),
            "confidence": float(max(probabilities)),
            "probabilities": probabilities,
        }

    except Exception as e:
        logging.exception("Inference error")
        return https_fn.HttpsError(
            https_fn.FunctionsErrorCode.INTERNAL,
            str(e)
        )


@https_fn.on_call(enforce_app_check=True)
def calc(req: https_fn.CallableRequest) :
    """
    Expected input (from Flutter):
    {
      "a": 5,
      "b": 3,
      "op": "add"
    }
    """

    data = req.data

    a = data.get("a")
    b = data.get("b")
    op = data.get("op")

    if a is None or b is None or op is None:
        return https_fn.HttpsError(
            https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            "Missing fields: a, b, op"
        )

    if op == "add":
        result = a + b
    elif op == "sub":
        result = a - b
    elif op == "mul":
        result = a * b
    elif op == "div":
        if b == 0:
            return https_fn.HttpsError(
                https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
                "Division by zero is not allowed"
            )
        result = a / b
    else:
        return https_fn.HttpsError(
            https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            f"Unknown operation '{op}'"
        )

    return {"result": result}

@https_fn.on_call()
def detect_fake_receipt(req: https_fn.CallableRequest):
    """
    HTTP Cloud Function for fake receipt detection
    Expects JSON with:
    - 'image': base64 encoded image
    - 'expected_fields': dict with expected values (e.g., {"total_amount": "rp14.000"})
    
    Returns JSON with verification results
    """
    data = req.data

    img_base64 = data.get("image")
    expected_fields = data.get("expected_fields", {})

    if not img_base64:
        raise https_fn.HttpsError(
            https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            "Missing 'image' field"
        )

    try:
        # Decode base64 into bytes
        img_bytes = base64.b64decode(img_base64)

        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise Exception("Failed to decode image")

        # Step 1: Text detection
        logging.info("Step 1: Text detection")
        result_img, boxes, gray, edges, dilated = process_text_detection(image)

        # Step 2: Line grouping
        logging.info("Step 2: Line grouping")
        lines = process_line_grouping(boxes)

        # Early validation: Line count check
        line_count_validation = validate_line_count(lines, min_lines=10, max_lines=20)
        
        if not line_count_validation["is_valid"]:
            return {
                "success": True,
                "early_detection": True,
                "verdict": "FAKE",
                "reason": line_count_validation["reason"],
                "details": {
                    "detected_lines": line_count_validation["detected_lines"],
                    "expected_range": line_count_validation["expected_range"]
                }
            }

        # Early validation: Layout structure check
        logging.info("Validating layout structure")
        layout_validation = validate_layout_structure(lines, image.shape[:2])
        
        if not layout_validation["is_valid"]:
            return {
                "success": True,
                "early_detection": True,
                "verdict": "FAKE",
                "reason": layout_validation["reason"],
                "details": {
                    "similarity_score": layout_validation["similarity_score"],
                    "threshold": layout_validation["threshold"],
                    "violations": layout_validation["violations"]
                }
            }

        # Step 3: Text extraction using both models
        logging.info("Step 3: Text extraction")
        lines_data = process_text_extraction(gray, lines)

        if not lines_data:
            return {
                "success": True,
                "verdict": "INCONCLUSIVE",
                "reason": "No text could be extracted from the image"
            }

        # Step 4: Verification using expected fields
        logging.info("Step 4: Verification")
        verification = process_verification(lines_data, expected_fields)

        # Prepare response
        response = {
            "success": True,
            "early_detection": False,
            "verdict": verification["combined"]["final_verdict"],
            "knn_pass": verification["combined"]["knn_pass"],
            "logistic_pass": verification["combined"]["logistic_pass"],
            "layout_validation": {
                "is_valid": layout_validation["is_valid"],
                "similarity_score": layout_validation["similarity_score"]
            },
            "line_count": len(lines_data),
            "extracted_fields": {}
        }

        # Add extracted fields from both models
        for line in lines_data:
            field_knn = line.get("field_knn")
            field_log = line.get("field_logistic")
            
            if field_knn and field_knn != "unknown":
                response["extracted_fields"][f"{field_knn}_knn"] = line.get("text_knn", "")
            
            if field_log and field_log != "unknown":
                response["extracted_fields"][f"{field_log}_logistic"] = line.get("text_logistic", "")

        # Add verification details
        response["verification_details"] = {
            "knn_checks": verification["knn_verification"],
            "logistic_checks": verification["logistic_verification"]
        }

        return response

    except Exception as e:
        logging.exception("Receipt detection error")
        raise https_fn.HttpsError(
            https_fn.FunctionsErrorCode.INTERNAL,
            f"Processing failed: {str(e)}"
        )

