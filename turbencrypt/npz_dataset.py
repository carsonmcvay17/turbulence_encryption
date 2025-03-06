import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# Custom Dataset for loading images and labels from a .npz file
class NPZDataset(Dataset):
    def __init__(self, npz_file, transform=None):
        # Load the .npz file
        self.data = np.load(npz_file, allow_pickle=True)
        
        # Assuming 'images' contains the image data (shape: [num_samples, height, width, channels])
        self.images = self.data['inputs']  # Load images
        self.labels = self.data['metadata']  # Load labels (shape: [num_samples])

        # If the images have 1 channel (grayscale), convert to 3 channels (RGB)
        if self.images.shape[-1] == 1:  # Check if the image has 1 channel
            self.images = np.repeat(self.images, 3, axis=-1)  # Replicate the grayscale channel 3 times
        
        
        
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