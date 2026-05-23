from init import init
from simple_extract import extract
from simple_preprocess import preprocess
from detect import detect
from corrected_recognize import recognize


img, _ = init()

frames = extract(img)

for i, frame in enumerate(frames):
    binary = preprocess(frame, i)

    detect(binary, frame_number=i)

recognize()