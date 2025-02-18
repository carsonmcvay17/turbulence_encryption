import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Any

import glob
from tqdm import tqdm

from turbencrypt.run_turbulence import Turbulence
from turbencrypt.make_forcing import Forcings


class movie():
    """
    class designed to create a movie of the turbulence simulations
    """

    def make_movie(self, image_dir: str, save_path: str, config: dict[str, Any], visualize_idx: int | None = None):
        """
        make movie
        """
        

        # define forcing images
        image_paths = glob.glob(f"{image_dir}/*")
        image_path = image_dir


        for idx, img_path in tqdm(enumerate(image_path), desc="Running sims...", total=len(image_path)):

            # define forcing function
            wave_number = 1
            offsets = ((0, 0), (0, 0))
            forcing_fun = lambda grid: Forcings().mod_kolmogorov_forcing(image_path, grid, k=wave_number, offsets=offsets)

            # run simulation
            model = Turbulence(**config)
            vorticity, forcing_x_array, forcing_y_array = model.run_turbulence(forcing_fun, movie=True)
            
            

        