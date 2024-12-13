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
from navier_stokes import NavierStokes2D2

class Dataset():
    def make_data():
        simulation_data = {}
        # define forcing_images
        forcing_images = [{"i": i, "description": f"image{i}"} for i in range(1,22)]

        for idx, imgs in enumerate(forcing_images):
            i = imgs["i"]
            img = f"/Users/carsonmcvay/desktop/gradschool/research/turbulence_encryption/raw_images/image_{i}.jpg"  # Ensure this is a string path
            grid = grids.Grid((256, 256), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
            viscosity = 1e-3

            # Debugging to ensure img is a string
            print(f"img(filepath): {img} (type:{type(img)})")

            # Correcting how the lambda function is passed
            wave_number = 1
            offsets = ((0, 0), (0, 0))

            # Use `Forcings().mod_kolmogorov_forcing` correctly by passing img as a string
            forcing_fn = lambda grid: Forcings().mod_kolmogorov_forcing(
                img, 
                grid, k=wave_number, offsets=offsets
            )

            # Instantiate the model with forcing_fn
            model = Turbulence()
            state_at_15, forcing_x_array, forcing_y_array = model.run_turbulence(
                forcing_fn  # Pass the forcing_fn to the model
            )

            forcing = jnp.stack((forcing_x_array, forcing_y_array), axis=-1)
            inputs_np = jnp.array(state_at_15)
            outputs_np = jnp.array(forcing)

            simulation_data[f"inputs_forcing_{idx}"] = inputs_np
            simulation_data[f"outputs_forcing_{idx}"] = outputs_np
            simulation_data[f"metadata_forcing{idx}"] = {
                "description": imgs["description"],
                "i": i,
            }

        # Save the data
        jnp.savez("multi_forcing_simulations.npz", **simulation_data)


            