import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Any

import glob
from tqdm import tqdm

from turbencrypt.run_turbulence import Turbulence
from turbencrypt.make_forcing import Forcings
from turbencrypt.data_utils import safe_standardize

class Dataset():
    def visualize_simulation(self, state, forcing_x, forcing_y, img_idx):
        """
        Visualize the simulation state at t=15 and the forcing function.
        Args:
            state: Simulation state at time t (array of shape [H, W]).
            forcing_x: Forcing function x-component (array of shape [H, W]).
            forcing_y: Forcing function y-component (array of shape [H, W]).
            img_idx: Index of the image being simulated.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(state, cmap="viridis", origin="lower")
        axes[0].set_title(f"State at time t (Image {img_idx})")
        axes[0].axis("off")

        axes[1].imshow(forcing_x, cmap="viridis", origin="lower")
        axes[1].set_title(f"Forcing X (Image {img_idx})")
        axes[1].axis("off")

        axes[2].imshow(forcing_y, cmap="viridis", origin="lower")
        axes[2].set_title(f"Forcing Y (Image {img_idx})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


    def make_data(self, image_dir: str, save_path: str, config: dict[str, Any], visualize_idx: int | None = None):
        """
        Generate dataset, optionally visualize one simulation
        I need to update how I'm loading img I want the path to be a variable 
        """
        inputs = [] # store simulations at t=15
        outputs = [] #store forcing functions
        metadata = []
        

        # define forcing images
        image_paths = glob.glob(f"{image_dir}/*")

        for idx, img_path in tqdm(enumerate(image_paths), desc="Running sims...", total=len(image_paths)):

            # define forcing function
            wave_number = 1
            offsets = ((0, 0), (0, 0))
            forcing_fun = lambda grid: Forcings().mod_kolmogorov_forcing(img_path, grid, k=wave_number, offsets=offsets)

            # run simulation
            model = Turbulence(**config)
            state, forcing_x_array, forcing_y_array = model.run_turbulence(forcing_fun)
            
            # normalize
            state_normalized = safe_standardize(state)
            forcing_x_normalized = safe_standardize(forcing_x_array)
            forcing_y_normalized = safe_standardize(forcing_y_array)

            # combine forcing
            forcing = jnp.stack((forcing_x_normalized, forcing_y_normalized), axis=-1)

             # Optionally visualize this simulation
            if visualize_idx is not None and idx == visualize_idx:
                self.visualize_simulation(
                    state_normalized, forcing_x_normalized, forcing_y_normalized, img_idx=idx
                )
                return  # Stop after visualizing


            # append
            inputs.append(state_normalized)
            outputs.append(forcing)
            metadata.append({"description": "NONE", "i": idx})

        # stack inputs and outputs into single arrays
        inputs = jnp.stack(inputs)
        outputs = jnp.stack(outputs)

        # Save the dataset
        jnp.savez(save_path, inputs=inputs, outputs=outputs, metadata=metadata)
        print(f"Dataset saved to {save_path}.")




    


            