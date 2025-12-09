from sklearn.base import BaseEstimator, TransformerMixin
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog
import numpy as np

class ImagePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, size=(64, 64)):
        self.size = size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = []
        for img in X:
            img_resized = resize(img, self.size, anti_aliasing=True)
            img_gray = rgb2gray(img_resized)
            out.append(img_gray)
        return np.array(out)

class HOGExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, pixels_per_cell=(8, 8)):
        self.pixels_per_cell = pixels_per_cell

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for img in X:
            feat = hog(img, pixels_per_cell=self.pixels_per_cell)
            features.append(feat)
        return np.array(features)
