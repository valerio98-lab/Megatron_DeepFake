import matplotlib.pyplot as plt
from megatron.video_dataloader import VideoDataset, VideoDataLoader

VIDEO_PATH = r"G:\My Drive\Megatron_DeepFake\dataset"
DEPTH_ANYTHING_SIZE = "Small"
NUM_FRAMES = 5
BATCH_SIZE = 2
SHUFFLE = True
dataset = VideoDataset(
    VIDEO_PATH, DEPTH_ANYTHING_SIZE, num_frame=NUM_FRAMES, num_video=BATCH_SIZE
)
dataloader = VideoDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
for batch in dataloader:
    print(len(batch))
    for elem in batch:
        for x in elem:
            print(len(x), end=" ")
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            plt.title("original" if x[2] else "manipulated")
            print(type(x[0]), type(x[1]), type(x[2]))
