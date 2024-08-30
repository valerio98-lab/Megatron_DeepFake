"""Definition of the Megatron Model"""

from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

import transformers  # type: ignore

from megatron.preprocessing import PositionalEncoding, RepVit
from megatron.utils import load_model
from megatron.video_dataloader import VideoDataset
from megatron.trans_one import TransformerFakeDetector


class Megatron:
    """Definition of the Megatron model"""

    def __init__(
        self,
        depth_anything_size: str = "Small",
        repvit_model: str = "repvit_m0_9.dist_300e_in1k",
        d_model: int = 384,
        num_frames=10,
        n_heads: int = 2,
        n_layers: int = 1,
        d_ff: int = 1024,
        num_classes: int = 2
    ):
        """
        Initializes an instance of the Megatron Model.

        Args:
            depth_anything_size (str, optional): The size of the depth-anything model. Defaults to "Small".
            repvit_model (str, optional): The RepVit model to use. Defaults to "repvit_m0_9.dist_300e_in1k".
            d_model (int, optional): The dimensionality of the model. Defaults to 384.
            max_len_pe (int, optional): The maximum length of the positional encoding. Defaults to 50.
            n_heads (int, optional): The number of attention heads. Defaults to 2.
            n_layers (int, optional): The number of transformer layers. Defaults to 2.
            d_ff (int, optional): The dimensionality of the feed-forward layer. Defaults to 1024.
            num_classes (int, optional): The number of output classes. Defaults to 2.
        """
        super().__init__()
        self.num_frames = num_frames
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, max_len=num_frames
        ).to(self.device)
        self.repvit = RepVit(repvit_model=repvit_model).to(self.device)
        self.model = TransformerFakeDetector(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            num_classes=num_classes,
        ).to(self.device)
        self.depth_anything = transformers.pipeline(
            task="depth-estimation",
            model=f"depth-anything/Depth-Anything-V2-{depth_anything_size}-hf",
            device=self.device,
        )

    def inference(
        self,
        video_path: Path,
        random_initial_frame: bool = False,
    ) -> str:
        """
        Perform inference on a video file to determine if it is fake or original.
        Args:
            video_path (Path): The path to the video file.
            num_frame (int, optional): The number of frames to use for inference. Defaults to 20.
            random_initial_frame (bool, optional): Whether to use a random initial frame for inference.
                Defaults to False.
        Returns:
            str: The result of the inference, either "fake" or "original".
        Raises:
            FileNotFoundError: If the video file is not found.
        """
        video = VideoDataset(
            video_path,
            depth_anything=self.depth_anything,
            num_video=1,
            num_frame=self.num_frames,
            random_initial_frame=random_initial_frame,
        )[0]
        self.repvit.eval()
        self.model.eval()
        if video is None:
            raise FileNotFoundError(f"Video file not found: {video_path}")
        with torch.no_grad():
            video.depth_frames = self.repvit(video.depth_frames.to(self.device))
            video.depth_frames = self.positional_encoding(video.depth_frames).unsqueeze(
                0
            )

            video.rgb_frames = self.repvit(video.rgb_frames.to(self.device))
            video.rgb_frames = self.positional_encoding(video.rgb_frames).unsqueeze(0)

            logits = self.model(video.depth_frames, video.rgb_frames)

            softmax = F.softmax(logits, dim=1)
            output = np.argmax(softmax[0].cpu().detach().numpy())

            return "fake" if output == 0 else "original"

    def from_pretrained(self, model_path: Path):
        """
        Loads a pretrained model from the specified `model_path`.

        Args:
            model_path (Path): The path to the pretrained model.

        Returns:
            None
        """
        load_model(self.model, model_path)
        return self
