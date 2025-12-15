import joblib
import logging
from constants.constants import BASE_DIR

MODEL_REGISTRY = {
    "simple_ml_classifier": BASE_DIR / "assets" / "image_classifier_logreg.pkl",
    "knn_ocr": BASE_DIR / "assets" / "knn_model.pkl",
    "logistic_regression_ocr": BASE_DIR / "assets" / "logistic_regression_model.pkl",
    "svm_ocr": BASE_DIR / "assets" / "svm_model.pkl",
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