from torchvision import io


def extract_frames():
    filepath = r"G:\My Drive\Megatron_DeepFake\dataset\manipulated_sequences\FaceShifter\raw\videos\033_097.mp4"
    frames = io.read_video(filename=filepath)
    print(frames)
