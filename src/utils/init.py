import cv2
import os
import shutil
import sys

from src.config import images, output_paths


def init():
    for path_key in ["output", "preprocess", "detect", "extract", "recognize", "words", "chars"]:
        path = output_paths[path_key]
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    if len(sys.argv) > 1:
        image_name = sys.argv[1]
        if not image_name.endswith(f".{images['format']}"):
            image_path = f"{images['path']}/{image_name}.{images['format']}"
        else:
            image_path = f"{images['path']}/{image_name}"
    else:
        image_path = None

    assert image_path is not None, "No image path entered"

    img = cv2.imread(image_path)
    cv2.imwrite(f"{output_paths['output']}/original.png", img)

    return img