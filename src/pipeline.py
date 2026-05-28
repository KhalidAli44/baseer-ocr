from src.utils.init import init
from src.extract.extract import extract
from src.preprocess.preprocess import preprocess
from src.detect.detect import detect
from src.recognize.recognize import recognize


def main():
    img, save_output = init()
    frames = extract(img, save_output=save_output)

    all_char_entries  = []
    all_word_line_map = {}

    for i, frame in enumerate(frames):
        binary = preprocess(frame, save_output=save_output, frame_number=i)
        char_entries, word_line_map = detect(binary, save_output=save_output, frame_number=i)
        all_char_entries.extend(char_entries)
        all_word_line_map.update(word_line_map)

    print(recognize(char_entries=all_char_entries, word_line_map=all_word_line_map, save_output=save_output))

if __name__ == "__main__":
    main()