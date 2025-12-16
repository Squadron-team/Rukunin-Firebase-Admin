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
from services.fake_receipt_detection.layout_aware_receipt_verification import (
    layout_aware_receipt_verification,
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


@https_fn.on_call(memory=1024)
def classify_image(req: https_fn.CallableRequest):
    """
    HTTP Cloud Function for image classification
    Expects JSON with 'image' field containing base64 encoded image
    Returns JSON with prediction results
    """
    data = req.data

    img_base64 = data.get("image")
    if not img_base64:
        raise https_fn.HttpsError(
            https_fn.FunctionsErrorCode.INVALID_ARGUMENT, "Missing 'image' field"
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
            raise https_fn.HttpsError(
                https_fn.FunctionsErrorCode.INTERNAL, "Model is not loaded properly!"
            )

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
        raise https_fn.HttpsError(https_fn.FunctionsErrorCode.INTERNAL, str(e))


@https_fn.on_call(enforce_app_check=True)
def calc(req: https_fn.CallableRequest):
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
        raise https_fn.HttpsError(
            https_fn.FunctionsErrorCode.INVALID_ARGUMENT, "Missing fields: a, b, op"
        )

    if op == "add":
        result = a + b
    elif op == "sub":
        result = a - b
    elif op == "mul":
        result = a * b
    elif op == "div":
        if b == 0:
            raise https_fn.HttpsError(
                https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
                "Division by zero is not allowed",
            )
        result = a / b
    else:
        raise https_fn.HttpsError(
            https_fn.FunctionsErrorCode.INVALID_ARGUMENT, f"Unknown operation '{op}'"
        )

    return {"result": result}


@https_fn.on_request(memory=1024) # type: ignore
def detect_fake_receipt(req: https_fn.Request): # type: ignore
    """
    HTTP Cloud Function for fake receipt detection
    - 'image': image from HTTP post request
    - 'expected_fields': string with expected values (e.g., "Rp14.000")

    Returns JSON with verification results
    """
    if "image" not in req.files:
        raise https_fn.HttpsError(
            https_fn.FunctionsErrorCode.INVALID_ARGUMENT, "No image file!"
        )

    expected_amount = req.form.get("expected_amount")

    if expected_amount is None:
        raise https_fn.HttpsError(
            https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            "Expected amount field is missing!",
        )

    try:
        file = req.files["image"]

        # Read bytes
        image_bytes = file.read()

        # Decode image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_np is None:
            raise https_fn.HttpsError(
                https_fn.FunctionsErrorCode.INVALID_ARGUMENT, "Invalid image"
            )
        verification, lines_data = layout_aware_receipt_verification(
            image=image_np,
            expected_amount={"total_amount": expected_amount.lower()},
        )

        return {
            "message": "success",
            "expected_amount": expected_amount,
            "final_verdict": verification.summary.final_verdict,
            "verification": verification,
            "lines_data": lines_data,
        }

    except Exception as e:
        logging.exception("Receipt detection error")
        raise https_fn.HttpsError(
            https_fn.FunctionsErrorCode.INTERNAL, f"Processing failed: {str(e)}"
        )
