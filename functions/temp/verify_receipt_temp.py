import logging
from dataclasses import dataclass
import numpy as np
from io import BytesIO
import cv2
import json
import joblib
from pathlib import Path
import base64
from PIL import Image
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from firebase_functions import https_fn
from werkzeug.wrappers import Response
from services import fake_receipt_detection_service
from typing import cast, Union

ModelType = Union[SVC, LogisticRegression, KNeighborsClassifier]


@dataclass
class MLModel:
    path: Path
    model: ModelType | None = None


_ml_models = {
    "simple_ml_classifier": MLModel(
        Path(__file__).parent / "assets" / "image_classifier_logreg.pkl"
    ),
    "knn_ocr": MLModel(Path(__file__).parent / "assets" / "knn_model.pkl"),
    "logistic_regression_ocrr": MLModel(
        Path(__file__).parent / "assets" / "logistic_regression_ocr.pkl"
    ),
    "svm_ocr": MLModel(Path(__file__).parent / "assets" / "svm_ocr.pkl"),
}

_models_loaded = False


def load_model() -> None:
    """Load all ML models once and cache them"""
    global _models_loaded
    if _models_loaded:
        return

    try:
        for name, m in _ml_models.items():
            logging.info(f"Loading model: {name}")
            m.model = joblib.load(m.path)
        _models_loaded = True
        logging.info("All models loaded successfully")
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise


def verify_receipt(req: https_fn.CallableRequest) -> Response:
    """
    HTTP Cloud Function for receipt verification
    Expects JSON with:
    - 'image': base64 encoded image
    - 'expected_amount': expected total amount string (optional)
    Returns JSON with verification results
    """
    data = req.data

    img_base64 = data.get("image")
    if not img_base64:
        # return {"success": False, "error": "Missing 'image' field"}
        return Response(
            json.dumps({"success": False, "error": "Missing 'image' field"}),
            mimetype="application/json",
            status=400,
        )

    expected_amount = data.get("expected_amount", None)

    try:
        # Decode base64 into bytes
        img_bytes = base64.b64decode(img_base64)

        # Open as PIL image and convert to numpy array (BGR for OpenCV)
        img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img_pil)
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Load models (cached)
        load_model()

        # Get models
        knn_model_raw = _ml_models["knn_ocr"].model
        logistic_model_raw = _ml_models["logistic_regression_ocr"].model

        if knn_model_raw is None or logistic_model_raw is None:
            raise Exception("Models not loaded properly")

        knn_model = cast(KNeighborsClassifier, knn_model_raw)
        logistic_model = cast(LogisticRegression, logistic_model_raw)

        # Prepare expected fields
        expected_fields = {}
        if expected_amount:
            expected_fields["total_amount"] = expected_amount.lower()

        # Run verification pipeline
        verification, lines_data = (
            fake_receipt_detection_service.verify_receipt_with_models(
                img_bgr, knn_model, logistic_model, expected_fields
            )
        )

        # Simplify response for Firebase
        response = {
            "success": True,
            "is_fake": not verification["combined"].get("knn_pass", False)
            and not verification["combined"].get("logistic_pass", False),
            "final_verdict": verification["combined"]["final_verdict"],
            "knn_pass": verification["combined"]["knn_pass"],
            "logistic_pass": verification["combined"]["logistic_pass"],
            "confidence": int(
                (
                    verification["combined"]["knn_pass"]
                    + verification["combined"]["logistic_pass"]
                )
                / 2
                * 100
            ),
            "early_detection": verification.get("early_detection", False),
            "detected_lines": len(lines_data),
        }

        # Add validation details if available
        if "line_validation" in verification:
            response["line_validation"] = {
                "is_valid": verification["line_validation"]["is_valid"],
                "detected_lines": verification["line_validation"]["detected_lines"],
                "reason": verification["line_validation"]["reason"],
            }

        if "layout_validation" in verification:
            response["layout_validation"] = {
                "is_valid": verification["layout_validation"]["is_valid"],
                "similarity_score": verification["layout_validation"][
                    "similarity_score"
                ],
                "reason": verification["layout_validation"]["reason"],
            }

        # return response
        return Response(json.dumps(response), mimetype="application/json", status=200)

    except Exception as e:
        logging.exception("Receipt verification error")

        # return {"success": False, "error": str(e)}
        return Response(
            json.dumps({"success": False, "error": str(e)}),
            mimetype="application/json",
            status=500,
        )
