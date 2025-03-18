# imports
import matplotlib.image as mpi
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import jax.numpy as jnp
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torch import nn 
from ignite.metrics import MaximumMeanDiscrepancy
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
data_path = "/Users/gilpinlab/turbulence_encryption/data/mnist_eval.npz"
data = jnp.load(data_path)

inputs = data['inputs']
targets = data['targets']
outputs = data['outputs']

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

inputs = torch.tensor(inputs).float().to(device)
targets = torch.tensor(targets).float().to(device)
outputs = torch.tensor(outputs).float().to(device)



model = models.resnet50(pretrained=True)
model = model.eval()
model = model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))



mmd = MMD()

mmd_values = []


batch_size = 1
for i in range(0, len(targets), batch_size):

    
    batch_targets = targets[i:i+batch_size].to(device)
    batch_outputs = outputs[i:i+batch_size].to(device)

    features_targets = mmd.extract_features(batch_targets, model)
    features_outputs = mmd.extract_features(batch_outputs, model)

    mmd_value = mmd.compute_mmd(features_targets, features_outputs, kernel = 'rbf')
    mmd_values.append(mmd_value.item())
    print(f"MMD value for batch {i//batch_size + 1}: {mmd_value.item()}")

# average_mmd = jnp.mean(mmd_values)
# print(f"Average MMD value: {average_mmd}")

# Visualize the distributions of x and y
plt.hist(features_targets.flatten().cpu().numpy(), bins=50, alpha=0.5, label='x (targets)')
plt.hist(features_outputs.flatten().cpu().numpy(), bins=50, alpha=0.5, label='y (outputs)')
plt.legend()
plt.show()





