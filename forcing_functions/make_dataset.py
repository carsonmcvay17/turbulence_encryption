import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import xarray
import json

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral

import dataclasses
import navier_stokes
from navier_stokes import NavierStokes2D2
from run_turbulence import Turbulence
from make_forcing import Forcings

class Dataset():
    def visualize_simulation(self, state_at_15, forcing_x, forcing_y, img_idx):
        """
        Visualize the simulation state at t=15 and the forcing function.
        Args:
            state_at_15: Simulation state at t=15 (array of shape [H, W]).
            forcing_x: Forcing function x-component (array of shape [H, W]).
            forcing_y: Forcing function y-component (array of shape [H, W]).
            img_idx: Index of the image being simulated.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(state_at_15, cmap="viridis", origin="lower")
        axes[0].set_title(f"State at t=15 (Image {img_idx})")
        axes[0].axis("off")

        axes[1].imshow(forcing_x, cmap="viridis", origin="lower")
        axes[1].set_title(f"Forcing X (Image {img_idx})")
        axes[1].axis("off")

        axes[2].imshow(forcing_y, cmap="viridis", origin="lower")
        axes[2].set_title(f"Forcing Y (Image {img_idx})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()


    def make_data(visualize_idx = None):
        """
        Generate dataset, optionally visualize one simulation
        """
        inputs = [] # store simulations at t=15
        outputs = [] #store forcing functions
        metadata = []

        # define forcing images
        forcing_images = [{"i": i, "description": f"image{i}"} for i in range(1,22)]

        for idx, imgs in enumerate(forcing_images):
            i = imgs["i"]
            img = f"/Users/carsonmcvay/desktop/gradschool/research/turbulence_encryption/raw_images/image_{i}.jpg"
            grid = grids.Grid((256,256), domain=((0,2*jnp.pi), (0,2*jnp.pi)))
            viscosity = 1e-2

            # define forcing function
            wave_number = 1
            offsets = ((0, 0), (0, 0))
            forcing_fun = lambda grid: Forcings().mod_kolmogorov_forcing(img,grid, k=wave_number, offsets=offsets)

            # run simulation
            model = Turbulence()
            state_at_15, forcing_x_array, forcing_y_array = model.run_turbulence(forcing_fun)

            # normalize
            state_at_15_normalized = (state_at_15 - jnp.mean(state_at_15)) / jnp.std(state_at_15)
            forcing_x_normalized = (forcing_x_array - jnp.mean(forcing_x_array)) / jnp.std(forcing_x_array)
            forcing_y_normalized = (forcing_y_array - jnp.mean(forcing_y_array)) / jnp.std(forcing_y_array)

            # combine forcing
            forcing = jnp.stack((forcing_x_normalized, forcing_y_normalized), axis=-1)

             # Optionally visualize this simulation
            if visualize_idx is not None and idx == visualize_idx:
                self.visualize_simulation(
                    state_at_15_normalized, forcing_x_normalized, forcing_y_normalized, img_idx=idx
                )
                return  # Stop after visualizing


            # append
            inputs.append(state_at_15_normalized)
            outputs.append(forcing)
            metadata.append({"description": imgs["description"], "i": i})

            print(f"simulation {idx + 1}/21 complete")

        # stack inputs and outputs into single arrays
        inputs = jnp.stack(inputs)
        outputs = jnp.stack(outputs)

        # Save the dataset
        save_path = "multi_forcing_simulations_combined_viscneg2.npz"
        jnp.savez(save_path, inputs=inputs, outputs=outputs, metadata=metadata)
        print(f"Dataset saved to {save_path}.")




    # def make_data():
    #     simulation_data = {}
    #     # define forcing_images
    #     forcing_images = [{"i": i, "description": f"image{i}"} for i in range(1,22)]

    #     for idx, imgs in enumerate(forcing_images):
    #         i = imgs["i"]
    #         img = f"/Users/carsonmcvay/desktop/gradschool/research/turbulence_encryption/raw_images/image_{i}.jpg"  # Ensure this is a string path
    #         grid = grids.Grid((256, 256), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
    #         viscosity = 1e-3

    #         # Debugging to ensure img is a string
    #         print(f"img(filepath): {img} (type:{type(img)})")

    #         # Correcting how the lambda function is passed
    #         wave_number = 1
    #         offsets = ((0, 0), (0, 0))

    #         # Use `Forcings().mod_kolmogorov_forcing` correctly by passing img as a string
    #         forcing_fn = lambda grid: Forcings().mod_kolmogorov_forcing(
    #             img, 
    #             grid, k=wave_number, offsets=offsets
    #         )

    #         # Instantiate the model with forcing_fn
    #         model = Turbulence()
    #         state_at_15, forcing_x_array, forcing_y_array = model.run_turbulence(
    #             forcing_fn  # Pass the forcing_fn to the model
    #         )

    #         forcing = jnp.stack((forcing_x_array, forcing_y_array), axis=-1)
    #         inputs_np = jnp.array(state_at_15)
    #         outputs_np = jnp.array(forcing)

    #         simulation_data[f"inputs_forcing_{idx}"] = inputs_np
    #         simulation_data[f"outputs_forcing_{idx}"] = outputs_np
    #         simulation_data[f"metadata_forcing{idx}"] = {
    #             "description": imgs["description"],
    #             "i": i,
    #         }

    #     # Save the data
    #     jnp.savez("multi_forcing_simulations.npz", **simulation_data)


            