import cv2
import matplotlib.pyplot as plt
import dlib


def extract_frames():
    filepath = r"G:\My Drive\Megatron_DeepFake\dataset\manipulated_sequences\FaceShifter\raw\videos\648_654.mp4"
    capture_card = cv2.VideoCapture(filepath)
    while True:
        ret, frame = capture_card.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        faces = detector(frame_gray)
        for i, face in enumerate(faces):
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_image = frame[y : y + h, x : x + w]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_image = cv2.resize(
                face_image, (face_image.shape[0] * 4, face_image.shape[1] * 4)
            )
            plt.imshow(face_image)
            plt.show()

    capture_card.release()


extract_frames()
