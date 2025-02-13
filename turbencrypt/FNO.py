# imports
import torch
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from turbencrypt.named_dataset import NamedTensorDataset
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.data.transforms.base_transforms import Transform
from neuralop.data.transforms.data_processors import DefaultDataProcessor
# from named_dataset import NamedTensorDataset


class StupidTransform(Transform):
    """
    UnitGaussianNormalizer normalizes data to be zero mean and unit std.
    """

    def __init__(self, min=None, max=None, eps=1e-7, dim=None):
        super().__init__()

        self.register_buffer("min", min)
        self.register_buffer("max", max)

        self.eps = eps

    def fit(self, data_batch):
        self.update_min_max(data_batch)

    def update_min_max(self, data_batch):
        self.min = data_batch.min()
        self.max = data_batch.max()

    def transform(self, x):
        return (x - self.min) / (self.max - self.min + self.eps)

    def inverse_transform(self, x):
        return x * (self.max - self.min + self.eps) + self.min

    def forward(self, x):
        return self.transform(x)

    def cuda(self):
        self.min = self.min.cuda()
        self.max = self.max.cuda()
        return self

    def cpu(self):
        self.min = self.min.cpu()
        self.max = self.max.cpu()
        return self

    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)
        return self

class FourierNO:
    """
    Create an FNO and train
    arguments:
    """


    def makeFNO(data, random_state, test_size):
        """
        Makes the FNO
        arguments: data-string to the data path
        random_state: int random seed
        test_size: float fraction<1
        """
        data = jnp.load(data)
        
        # define imputs and outputs
        inputs = data["inputs"]
        outputs = data["outputs"]


        # training and test split
        # split
        X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=test_size, random_state=random_state)

        # convert data from jnp to np
        # Convert data from JAX arrays to NumPy
        # doing this before passing to PyTorch
        X_train_np = np.array(jnp.asarray(X_train).block_until_ready())  # Ensures conversion to NumPy
        y_train_np = np.array(jnp.asarray(y_train).block_until_ready())
        X_test_np = np.array(jnp.asarray(X_test).block_until_ready())
        y_test_np = np.array(jnp.asarray(y_test).block_until_ready())

        # data loaders
        # data loaders
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).permute(0, 3, 1, 2)
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).unsqueeze(1)
        y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).permute(0, 3, 1, 2)

        encoder_in = StupidTransform()
        encoder_in.fit(X_train_tensor)
        encoder_out = UnitGaussianNormalizer()
        encoder_out.fit(y_train_tensor)

        # create DataProcessor
        data_processor = DefaultDataProcessor(
            in_normalizer=encoder_in,
            out_normalizer=encoder_out,
        )

        # Create data loaders
        # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_dataset = NamedTensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = NamedTensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        return train_loader, test_loader, data_processor



