import numpy as np
import jax.numpy as jnp

def check_nans():
    data_path = "/Users/carsonmcvay/desktop/gradschool/research/turbulence_encryption/forcing_functions/multi_forcing_simulations_combined_viscneg2.npz"
    data = np.load(data_path)
    for key in data:
        if np.isnan(data[key]).any():
            print(f"NaNs found in {key}")
            print(f"{np.isnan(data[key]).sum()}/{data[key].size} NaNs")
            exit()
    print("No NaNs")


check_nans()