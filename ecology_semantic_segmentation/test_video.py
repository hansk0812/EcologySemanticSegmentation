from collections import OrderedDict
import torch
import torchvision.transforms as transforms
from torchvision import models, utils
import numpy as np
from PIL import Image
import os
import cv2
from torch.utils.data import Dataset
import argparse

from ecology_semantic_segmentation.test_multiclass_sequential_densenetloss import unet_model
# from ecology_semantic_segmentation.dataset.fish.video_dataset import VideoDataset, parse_video
utils.save_image

def parse_video(file_source, output_dir="./frames"):
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(output_dir, f"frame_{i}.jpg"), frame)
    
    # Release the video capture object
    video_capture.release()
    
    # Convert the frames list to a numpy array
    frames = np.array(frames)

    # Convert the labels list to a numpy array (if applicable)
    # labels = np.array(labels)

    return frames

class VideoDataset(Dataset):
    def __init__(self, file_source):
        self.frames, self.labels = parse_video(file_source)
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx] if self.labels else None
        return frame, label


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def load_model(model_path):
    # Load the model checkpoint using torch.load, which returns an OrderedDict
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    # Check if the checkpoint is an OrderedDict
    if isinstance(checkpoint, OrderedDict):
        # Assuming your model is named 'model' inside the checkpoint
        model = checkpoint['model']
        # Load the state dictionary of the model
        model.load_state_dict(checkpoint['state_dict'])
        return model
    else:
        raise TypeError("The loaded checkpoint is not in the expected format.")


if __name__ == "__main__":
    # Load the pre-trained model
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Path to the pre-trained model")
    ap.add_argument("--video_path", default="input", help="Path to the input video")
    ap.add_argument("--result_dir", default="results", help="Directory to save the results")
    args = ap.parse_args()
    # if they did not provide the model_path and the result directory, tell them to provide it
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at {args.model_path}")
    # if the argument was not passed give them a warning
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    if torch.cuda.is_available():
        net = unet_model.cuda()
        net.load_state_dict(torch.load(args.model_path))
    else:
        net = unet_model.cpu()
        net.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))
    
    net = net.eval()

    
    # load the video file and parse it into individual frames
    video_dataset = VideoDataset(args.video_path)
    video_dataloader = torch.utils.data.DataLoader(video_dataset, batch_size=1, shuffle=False)
    
    # Process each frame in the video
    for i, (frame, _) in enumerate(video_dataloader):
        # Perform inference on the frame
        with torch.no_grad():
            output = net(frame)
        
        # Save the output image
        output_image = output.squeeze().numpy()
        output_image = (output_image * 255).astype(np.uint8)
        output_image = Image.fromarray(output_image)
        output_image.save(os.path.join(args.result_dir, f"output_frame_{i}.jpg"))