from src.utils.init import init
from src.extract.extract import extract
from src.preprocess.preprocess import preprocess
from src.detect.detect import detect
from src.recognize.recognize import recognize


def main():
    img = init()
    frames = extract(img)

    for i, frame in enumerate(frames):
        binary = preprocess(frame, i)
        detect(binary, frame_number=i)

    recognize()


if __name__ == "__main__":
    main()