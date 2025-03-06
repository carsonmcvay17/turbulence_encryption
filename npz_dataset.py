import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Custom Dataset for loading images and labels from a .npz file
class NPZDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        # Load the .npz file
        self.data = np.load(npz_file)
        
        # Assuming 'images' contains the image data (shape: [num_samples, height, width, channels])
        self.images = self.data['images']  # Load images
        self.labels = self.data['labels']  # Load labels (shape: [num_samples])
        
        self.transform = transform  # Any transformation to apply to images

    def __len__(self):
        # Return the number of samples
        return len(self.images)

    def __getitem__(self, idx):
        # Get the image and label by index
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert the image from numpy array to PIL image
        image = Image.fromarray(image)
        
        # Apply the transformation (e.g., resize, normalize)
        if self.transform:
            image = self.transform(image)
        
        return image, label