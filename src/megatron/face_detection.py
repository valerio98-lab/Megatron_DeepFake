import cv2 as cv
import matplotlib.pyplot as plt


def extract_frames():
    filepath = r"G:\My Drive\Megatron_DeepFake\dataset\manipulated_sequences\FaceShifter\raw\videos\033_097.mp4"
    capture_card = cv.VideoCapture(filepath)
    while True:
        ret, frame = capture_card.read()
        if not ret:
            break
        plt.imshow(frame)
        plt.show()
    capture_card.release()


extract_frames()
