# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app
import joblib
import base64
import logging
from pathlib import Path
from PIL import Image
from io import BytesIO
import numpy as np
import sys
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from services.image_processing import ImagePreprocessor, HOGExtractor
from typing import Union

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

# Global variable to cache the model
ModelType = Union[SVC, LogisticRegression, KNeighborsClassifier]

MODEL_REGISTRY = {
    "simple_ml_classifier": Path(__file__).parent / "assets" / "image_classifier_logreg.pkl",
    "knn_ocr": Path(__file__).parent / "assets" / "knn_model.pkl",
    "logistic_regression_ocr": Path(__file__).parent / "assets" / "logistic_regression_ocr.pkl",
    "svm_ocr": Path(__file__).parent / "assets" / "svm_ocr.pkl",
}

_MODEL_CACHE = {}

def get_model(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name not in _MODEL_CACHE:
        logging.info(f"Loading model: {model_name}")
        _MODEL_CACHE[model_name] = joblib.load(MODEL_REGISTRY[model_name])
        logging.info(f"Model {model_name} loaded")

    return _MODEL_CACHE[model_name]

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

