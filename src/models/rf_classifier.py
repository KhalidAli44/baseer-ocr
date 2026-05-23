import pickle
import numpy as np

from features import extract
from config import RF_MODEL_PATH, RF_SCALER_PATH, RF_ENCODER_PATH, TOP_K


class RFClassifier:
    def __init__(self):
        with open(RF_MODEL_PATH, "rb") as f:
            self._clf = pickle.load(f)
        with open(RF_SCALER_PATH, "rb") as f:
            self._scaler = pickle.load(f)
        with open(RF_ENCODER_PATH, "rb") as f:
            self._encoder = pickle.load(f)
    
    def predict(self, char_img):
        feat = extract(char_img).reshape(1, -1)
        scaled = self._scaler.transform(feat)
        probs = self._clf.predict_proba(scaled)[0]
        top_idx = np.argsort(probs)[::-1][:TOP_K]
        return [(self._encoder.inverse_transform([i])[0], float(probs[i])) for i in top_idx]
    
    def predict_best(self, char_img):
        candidates = self.predict(char_img)
        return candidates[0]
    
    def predict_batch(self, char_imgs):
        from features import extract_batch
        feats = extract_batch(char_imgs)
        scaled = self._scaler.transform(feats)
        probs = self._clf.predict_proba(scaled)
        results = []
        for row in probs:
            top_idx = np.argsort(row)[::-1][:TOP_K]
            results.append([(self._encoder.inverse_transform([i])[0], float(row[i])) for i in top_idx])
        return results