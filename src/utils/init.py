import cv2
import os
import shutil
import sys

from config import OUTPUT_DIR, PREPROCESS_DIR, DETECT_DIR, EXTRACT_DIR, WORDS_DIR, CHARS_DIR, TEST_IMAGES_DIR, IMAGE_FORMAT

def init():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    if os.path.exists(PREPROCESS_DIR):
        shutil.rmtree(PREPROCESS_DIR)
    os.makedirs(PREPROCESS_DIR)
    if os.path.exists(DETECT_DIR):
        shutil.rmtree(DETECT_DIR)
    os.makedirs(DETECT_DIR)
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    os.makedirs(EXTRACT_DIR)
    if os.path.exists(WORDS_DIR):
        shutil.rmtree(WORDS_DIR)
    os.makedirs(WORDS_DIR)
    if os.path.exists(CHARS_DIR):
        shutil.rmtree(CHARS_DIR)
    os.makedirs(CHARS_DIR)

    if len(sys.argv) > 1:
        image_name = sys.argv[1]
        if not image_name.endswith(f".{IMAGE_FORMAT}"):
            image_path = f"{TEST_IMAGES_DIR}/{image_name}.{IMAGE_FORMAT}"
        else:
            image_path = f"{TEST_IMAGES_DIR}/{image_name}"
    assert image_path is not None, "No image path entered"
    img = cv2.imread(image_path)

    is_nat = (sys.argv[2] == "nat") if len(sys.argv) > 2 else False

    cv2.imwrite(f"{OUTPUT_DIR}/original.png",  img)

    return img, is_nat