import numpy as np
import jax.numpy as jnp
from turbencrypt.data_utils import nan_check

# def check_nans():
#     data_path = "/Users/carsonmcvay/desktop/gradschool/research/turbulence_encryption/data/test_sim_visc1.npz"
#     data = np.load(data_path)
#     for key in data:
#         if np.isnan(data[key]).any():
#             print(f"NaNs found in {key}")
#             print(f"{np.isnan(data[key]).sum()}/{data[key].size} NaNs")
#             exit()
#     print("No NaNs")



data = "/Users/carsonmcvay/desktop/gradschool/research/turbulence_encryption/data/test_sim_visc1.npz"
nan_check(data)