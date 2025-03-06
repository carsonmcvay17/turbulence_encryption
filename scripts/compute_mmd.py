# imports
import matplotlib.image as mpi
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torch import nn 
from turbencrypt.mmd import MMD
from turbencrypt.FNO import FourierNO
from turbencrypt.npz_dataset import NPZDataset


# run the mmd on the different data sets
# okay this is very bad and does not work

# Step 1: Define data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),   # Resize image to the size the model expects
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize based on ImageNet stats
])

# load the data
data_path1 = "/Users/gilpinlab/turbulence_encryption/data/mnist_dataset.npz"

dataset = NPZDataset(data_path1, transform=transform)





data_loader1 = DataLoader(dataset, batch_size=32, shuffle=True)
data_loader2 = DataLoader(dataset, batch_size=32, shuffle=True)

model = models.resnet50(pretrained=True)
model = model.eval()
model = model.to(torch.devie("cuda") if torch.cuda.is_available() else torch.device("cpu"))

mmd = MMD()


features1 = mmd.extract_features(data_loader1, model)
features2 = mmd.extract_features(data_loader2, model)

mmd_value = mmd.compute_mmd(features1, features2, kernel="rbf")
print(f"MMD value: {mmd_value}")




