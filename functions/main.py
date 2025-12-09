# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn
from firebase_functions.options import set_global_options
from firebase_admin import initialize_app
import joblib
import json
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


@https_fn.on_request()
def classify_image(req: https_fn.Request) -> https_fn.Response:
    """
    HTTP Cloud Function for image classification
    Expects JSON with 'image' field containing base64 encoded image
    Returns JSON with prediction results
    """
    # Set CORS headers
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    # Handle preflight requests
    if req.method == "OPTIONS":
        return https_fn.Response("", headers=headers)

    # Only accept POST requests
    if req.method != "POST":
        return https_fn.Response(
            json.dumps({"error": "Only POST requests are accepted"}),
            status=405,
            headers=headers,
            mimetype="application/json",
        )

    try:
        # Get uploaded file
        uploaded_file = req.files.get("image")
        if not uploaded_file:
            return https_fn.Response(
                json.dumps({"error": "Missing file field 'image'"}),
                status=400,
                headers=headers,
                mimetype="application/json",
            )

        img_bytes = uploaded_file.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)

        model = load_model()

        prediction = model.predict([img_np])[0]
        probabilities = model.predict_proba([img_np])[0].tolist()

        resp = {
            "prediction": int(prediction),
            "probabilities": probabilities,
            "confidence": float(max(probabilities)),
            "success": True,
        }

        return https_fn.Response(
            json.dumps(resp),
            status=200,
            headers=headers,
            mimetype="application/json",
        )

    except Exception as e:
        logging.exception("Inference error")
        return https_fn.Response(
            json.dumps({"error": str(e), "success": False}),
            status=500,
            headers=headers,
            mimetype="application/json",
        )


@https_fn.on_request()
def calc(req: https_fn.Request) -> https_fn.Response:
    """
    Expected JSON body:
    {
      "a": 5,
      "b": 3,
      "op": "add" | "sub" | "mul" | "div"
    }
    """
    if req.method == "GET":
        return https_fn.Response(
            json.dumps(
                {"message": "Calculator API is running. Use POST with JSON body."}
            ),
            mimetype="application/json",
        )

    data = req.get_json(silent=True)
    if not data:
        return https_fn.Response(
            json.dumps({"error": "JSON body required"}),
            status=400,
            mimetype="application/json",
        )

    a = data.get("a")
    b = data.get("b")
    op = data.get("op")

    if a is None or b is None or op is None:
        return https_fn.Response(
            json.dumps({"error": "Missing fields: a, b, op"}),
            status=400,
            mimetype="application/json",
        )

    if op == "add":
        result = a + b
    elif op == "sub":
        result = a - b
    elif op == "mul":
        result = a * b
    elif op == "div":
        if b == 0:
            return https_fn.Response(
                json.dumps({"error": "Division by zero"}),
                status=400,
                mimetype="application/json",
            )
        result = a / b
    else:
        return https_fn.Response(
            json.dumps({"error": f"Unknown operation '{op}'"}),
            status=400,
            mimetype="application/json",
        )

    return https_fn.Response(
        json.dumps({"result": result}), mimetype="application/json"
    )
