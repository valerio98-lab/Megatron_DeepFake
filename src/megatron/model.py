import os
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


from megatron.preprocessing import PositionalEncoding, RepVit

# from megatron.utils import save_checkpoint, save_model
import transformers
from megatron.video_dataloader import VideoDataset
from megatron.trans_one import TransformerFakeDetector
from megatron import DEVICE


# TODO: Valerio, il fakedetector e' stato modificato, fare i check e controlla che funziona tutto :)
class Megatron:
    def __init__(
        self,
        depth_anything_size: str = "Small",
        repvit_model: str = "repvit_m0_9.dist_300e_in1k",
        d_model: int = 384,
        max_len_pe=50,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 1024,
        num_classes: int = 2,
    ):
        super().__init__()
        self.positional_encoder = PositionalEncoding(
            d_model=d_model, max_len=max_len_pe
        )
        self.repvit = RepVit(repvit_model=repvit_model).eval()
        self.model = TransformerFakeDetector(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            num_classes=num_classes
        ).eval()
        self.depth_anything = transformers.pipeline(
            task="depth-estimation",
            model=f"depth-anything/Depth-Anything-V2-{depth_anything_size}-hf",
            device=DEVICE,
        )

    def inference(
        self,
        video_path: str,
        num_frame: int = 20,
        random_initial_frame: bool = False,
        num_video: int = 1,
    ):
        assert os.path.exists(video_path), f"Video path {video_path} does not exist"

        video = VideoDataset(
            video_path,
            depth_anything=self.depth_anything,
            num_video=1,
            num_frame=num_frame,
            random_initial_frame=random_initial_frame,
        )[0]

        video.depth_frames = self.repvit(video.depth_frames.to(DEVICE))
        video.depth_frames = self.positional_encoder(video.depth_frames).unsqueeze(0)

        video.rgb_frames = self.repvit(video.rgb_frames.to(DEVICE))
        video.rgb_frames = self.positional_encoder(video.rgb_frames).unsqueeze(0)

        logits = self.model(video.depth_frames, video.rgb_frames)

        softmax = F.softmax(logits, dim=1)
        output = np.argmax(softmax[0].detach().numpy())

        return "fake" if output == 0 else "original"


    def from_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model"])
        self.positional_encoder.load_state_dict(checkpoint["positional_encoder"])


if __name__ == "__main__":
    print("Ciao")
    megatron = Megatron(depth_anything_size="Small", n_layers=1)
    path = "/Users/valerio/Desktop/ao/dai"
    output = megatron.inference(video_path=path, num_frame=5)

    print(output)

    # for root, _, files in os.walk(path):
    #     print("son dentro")
    #     print(root, files)

    # for elem in batch:
    #     print(elem)
    #     print(len(elem))
    #     print(elem[0].shape, elem[1].shape, elem[2].shape)
    #     print(elem[0].dtype, elem[1].dtype, elem[2].dtype)
    #     print(type(elem))
