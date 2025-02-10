# imports
import torch
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from turbencrypt.named_dataset import NamedTensorDataset
# from named_dataset import NamedTensorDataset


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
        inputs = jnp.fft.irfft(data["inputs"], axis=-1)
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
        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)

        # Create data loaders
        # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_dataset = NamedTensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = NamedTensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, test_loader



