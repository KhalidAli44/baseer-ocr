from init import init
from extract import extract
from preprocess import preprocess
from detect import detect
from corrected_recognize import recognize


img, _ = init()

frames = extract(img)

for i, frame in enumerate(frames):
    binary = preprocess(frame, frame_number=i)

    detect(binary, frame_number=i)

recognize()