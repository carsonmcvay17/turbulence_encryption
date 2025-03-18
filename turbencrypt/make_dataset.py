import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Any

import glob
from tqdm import tqdm

from turbencrypt.run_turbulence import Turbulence
from turbencrypt.make_forcing import Forcings, FourierTransform
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


    def run_sim_from_images(self, image_dir: str, save_path: str, config: dict[str, Any]):
        """
        Generate dataset, optionally visualize one simulation
        """
        fftmodel = FourierTransform()
        image_paths = glob.glob(f"{image_dir}/*")
        image_iter = (fftmodel.load_image(img_path) for img_path in image_paths)
        return self.run_sim_from_tensors(image_iter, save_path, config, length=len(image_paths)) 


    def run_sim_from_tensors(self, tensors: list[jnp.ndarray], save_path: str, config: dict[str, Any], length: int = None):
        inputs = [] # store simulations at t=15
        outputs = [] #store forcing functions
        metadata = []
        length = length or len(tensors)
        for idx, field in tqdm(enumerate(tensors), desc="Running sims...", total=length):
            vorticity_normalized, forcing = self.run_single_sim(field, config)
        
            inputs.append(vorticity_normalized)
            outputs.append(forcing)
            metadata.append({"description": "NONE", "i": idx})
            

        # stack inputs and outputs into single arrays
        inputs = jnp.stack(inputs)
        outputs = jnp.stack(outputs)

        jnp.savez(save_path, inputs=inputs, outputs=outputs, metadata=metadata)
        print(f"Dataset saved to {save_path}.")

    def run_single_sim(self, img: jnp.ndarray, config):
        # define forcing function
        wave_number = 1
        offsets = ((0, 0), (0, 0))
        forcing_fun = lambda grid: Forcings().mod_kolmogorov_forcing(img, grid, offsets=offsets)

        # run simulation
        model = Turbulence(**config)
        vorticity, forcing_x_array, forcing_y_array = model.run_turbulence(forcing_fun)
        
        # if forcing_x_array.ndim == 2:
        #     forcing_x_array = forcing_x_array[..., None]
        # if vorticity.ndim == 2:
        #     vorticity = vorticity[..., None]

        # # normalize
        # vorticity_normalized = safe_standardize(vorticity).squeeze()
        # forcing_x_normalized = safe_standardize(forcing_x_array).squeeze()
        # breakpoint()

        return vorticity, forcing_x_array



    


            