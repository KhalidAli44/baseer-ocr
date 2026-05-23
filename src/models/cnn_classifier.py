import torch
import pickle
import numpy as np
import cv2
from train_cnn import CNN
from config import CNN_MODEL_PATH, CNN_ENCODER_PATH, TOP_K

NORM_SIZE = 28


class CNNClassifier:
    def __init__(self):
        with open(CNN_ENCODER_PATH, "rb") as f:
            self._encoder = pickle.load(f)
        
        num_classes = len(self._encoder.classes_)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = CNN(num_classes).to(self._device)
        self._model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=self._device))
        self._model.eval()
    
    def _normalize(self, char_img):
        """Resize and center the character image to fixed size"""
        h, w = char_img.shape
        if h == 0 or w == 0:
            return np.ones((NORM_SIZE, NORM_SIZE), dtype=np.float32) * 255
        
        scale = (NORM_SIZE - 4) / max(h, w)
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        scaled = cv2.resize(char_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        canvas = np.ones((NORM_SIZE, NORM_SIZE), dtype=np.float32) * 255
        y_off = (NORM_SIZE - new_h) // 2
        x_off = (NORM_SIZE - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = scaled
        
        return canvas
    
    def _preprocess(self, char_img):
        """Preprocess single image for CNN input"""
        norm = self._normalize(char_img)
        tensor = torch.FloatTensor(norm).unsqueeze(0).unsqueeze(0) / 255.0
        return tensor.to(self._device)
    
    def predict(self, char_img):
        tensor = self._preprocess(char_img)
        with torch.no_grad():
            outputs = self._model(tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        top_idx = np.argsort(probs)[::-1][:TOP_K]
        return [(self._encoder.inverse_transform([i])[0], float(probs[i])) for i in top_idx]
    
    def predict_best(self, char_img):
        candidates = self.predict(char_img)
        return candidates[0]
    
    def predict_batch(self, char_imgs):
        """Process batch of images with different sizes"""
        normalized = []
        for img in char_imgs:
            norm = self._normalize(img)
            normalized.append(norm)
        
        batch = np.array(normalized)
        batch = torch.FloatTensor(batch).unsqueeze(1) / 255.0
        batch = batch.to(self._device)
        
        with torch.no_grad():
            outputs = self._model(batch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        results = []
        for row in probs:
            top_idx = np.argsort(row)[::-1][:TOP_K]
            results.append([(self._encoder.inverse_transform([i])[0], float(row[i])) for i in top_idx])
        return results