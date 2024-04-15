import cv2
import os
import numpy as np
from torch.utils.data import Dataset

def parse_video(file_source):
    frames = []
    labels = []  # If you have labels associated with each frame, you can store them here
    
    # Open the video file
    video_capture = cv2.VideoCapture(file_source)

    # Loop through each frame
    while True:
        # Read the frame
        ret, frame = video_capture.read()

        # Break the loop when the video is over
        if not ret:
            break
        
        # Append the frame to the frames list
        frames.append(frame)
        

        # If you have labels associated with each frame, you can append them here
        # For example, you can read labels from a separate file or extract them from the frame
        
        # labels.append(label)
    # Store the frames into its own directory
    if not os.path.exists("./frames"):
        os.makedirs("./frames")
    for i, frame in enumerate(frames):
        cv2.imwrite("./frames/frame%d.jpg" % i, frame)
    
    # Release the video capture object
    video_capture.release()
    
    # Convert the frames list to a numpy array
    frames = np.array(frames)

    # Convert the labels list to a numpy array (if applicable)
    # labels = np.array(labels)

    return frames, labels

class VideoDataset(Dataset):
    def __init__(self, file_source):
        self.frames, self.labels = parse_video(file_source)
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx] if self.labels else None
        return frame, label
