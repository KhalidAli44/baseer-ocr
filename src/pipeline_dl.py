from src.dl.inference import infer
from src.utils.init import init


def main():
    img, save_output = init()

    print(infer(img, save_output=save_output))

if __name__ == "__main__":
    main()