import os 
import cv2
from pathlib import Path

import torch 
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, video_dir, num_frame=1):
        super().__init__()
        self.video_dir = video_dir
        self.num_frame = num_frame 
        self.video_folder = Path(video_dir)
        self.cap = cv2.VideoCapture(str(self.video_folder))
        self.length = min(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), num_frame)
        
        assert (
            self.video_folder.exists()
        ), f'Watch out! "{str(self.video_folder)}" was not found.'
    
    def __len__(self):
        return self.length
    
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def __getitem__(self, idx):
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx) 
            ret, frame = self.cap.read() # ret is a boolean value that indicates if the frame was read correctly or not. frame is the image in BGR format.
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #face cropping operations
                face_crop = self.face_extraction(frame)
                #depth map operations on face_crop
                depth_mask = self.calculate_depth_mask(face_crop) 

                return face_crop, depth_mask
                

    def face_extraction(self, frame):
        #face cropping operations
        pass

    def calculate_depth_mask(self, face_crop):
        #depth map operations on face_crop
        pass