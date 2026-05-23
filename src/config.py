OUTPUT_DIR      = "output"
PREPROCESS_DIR  = f"{OUTPUT_DIR}/preprocess"
DETECT_DIR      = f"{OUTPUT_DIR}/detect"
EXTRACT_DIR     = f"{OUTPUT_DIR}/extract"
WORDS_DIR       = f"{OUTPUT_DIR}/words"
CHARS_DIR       = f"{OUTPUT_DIR}/chars"
TEST_IMAGES_DIR = "phone_images"
IMAGE_FORMAT    = "jpg"

MODELS_DIR   = "models/classical"
EMNIST_DIR   = "data/emnist"
DAWG_PATH = "models/classical/langmodel/dawg.pkl"
MODEL_PATH   = f"{MODELS_DIR}/SVM/classifier.pkl"
SCALER_PATH  = f"{MODELS_DIR}/SVM/scaler.pkl"
ENCODER_PATH = f"{MODELS_DIR}/SVM/encoder.pkl"
TOP_K        = 5

CNN_MODEL_PATH   = f"{MODELS_DIR}/CNN/cnn_model.pth"
CNN_ENCODER_PATH = f"{MODELS_DIR}/CNN/cnn_encoder.pkl"

RF_MODEL_PATH   = f"{MODELS_DIR}/RF/rf_model.pkl"
RF_SCALER_PATH  = f"{MODELS_DIR}/RF/rf_scaler.pkl"
RF_ENCODER_PATH = f"{MODELS_DIR}/RF/rf_encoder.pkl"

AH_HEIGHT_MIN      = 0.3
AH_HEIGHT_MAX      = 4.5
AH_AREA_MIN        = 0.05