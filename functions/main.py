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
from services.image_processing import ImagePreprocessor, HOGExtractor


# Fix for loading pickled scikit-learn pipelines:
# The model was trained in an environment where custom transformers
# (ImagePreprocessor, HOGExtractor) were defined under the __main__ module.
# Pickle stores class locations, so we alias these classes into __main__
# to ensure joblib.load() can reconstruct the pipeline correctly.
sys.modules['__main__'].ImagePreprocessor = ImagePreprocessor
sys.modules['__main__'].HOGExtractor = HOGExtractor

# For cost control, you can set the maximum number of containers that can be
# running at the same time. This helps mitigate the impact of unexpected
# traffic spikes by instead downgrading performance. This limit is a per-functionfir
# limit. You can override the limit for each function using the max_instances
# parameter in the decorator, e.g. @https_fn.on_request(max_instances=5).
set_global_options(max_instances=2)

initialize_app()

# Global variable to cache the model
_model = None
# Use absolute path relative to the functions directory
_model_path = Path(__file__).parent / "assets" / "image_classifier_linreg.pkl"

def load_model():
    global _model
    if _model is None:
        logging.info(f"Loading model from: {_model_path}")
        if not _model_path.exists():
            raise FileNotFoundError(f"Model not found at {_model_path}")
        _model = joblib.load(_model_path)
    return _model

@https_fn.on_call()
def classify_image(req: https_fn.CallableRequest) -> https_fn.Response:
    """
    HTTP Cloud Function for image classification
    Expects JSON with 'image' field containing base64 encoded image
    Returns JSON with prediction results
    """
    data = req.data

    img_base64 = data.get("image")
    if not img_base64:
        return {"success": False, "error": "Missing 'image' field"}

    try:
        # Decode base64 into bytes
        img_bytes = base64.b64decode(img_base64)

        # Open as PIL image
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        # Load model (cached)
        model = load_model()

        # Predict
        prediction = model.predict([img_np])[0]
        probabilities = model.predict_proba([img_np])[0].tolist()

        return {
            "success": True,
            "prediction": int(prediction),
            "confidence": float(max(probabilities)),
            "probabilities": probabilities
        }

    except Exception as e:
        logging.exception("Inference error")
        return {"success": False, "error": str(e)}


@https_fn.on_call()
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
        return {"error": "Missing fields: a, b, op"}

    if op == "add":
        result = a + b
    elif op == "sub":
        result = a - b
    elif op == "mul":
        result = a * b
    elif op == "div":
        if b == 0:
            return {"error": "Division by zero"}
        result = a / b
    else:
        return {"error": f"Unknown operation '{op}'"}

    return {"result": result}
