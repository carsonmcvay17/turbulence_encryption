import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Any

import glob
from tqdm import tqdm
import matplotlib.image as mpi
from skimage.transform import resize
from jax_cfd.base.forcings import kolmogorov_forcing

from turbencrypt.run_turbulence import Turbulence
from turbencrypt.make_forcing import Forcings
from turbencrypt.make_dataset import Dataset
from turbencrypt.make_forcing import Forcings, FourierTransform


class movie():
    """
    class designed to create a movie of the turbulence simulations
    """

    def make_movie(self, image_dir: str, save_path: str, sim_config: dict[str, Any], visualize_idx: int | None = None):
        """
        make movie
        """
        
        # run simulation
        image_path = image_dir
        fftmodel = FourierTransform()
        img = fftmodel.load_image(image_path) 
        # breakpoint()
        # forcing_fn = lambda grid: Forcings().mod_kolmogorov_forcing(img, grid)
        forcing_fn = lambda grid: kolmogorov_forcing(grid, k=6)
        model = Turbulence(**sim_config)
        vorticity, forcing_x_array, forcing_y_array = model.run_turbulence(forcing_fn, movie=True)
        # define forcing images
        # image_paths = glob.glob(f"{image_dir}/*")
    


        # for idx, img_path in tqdm(enumerate(image_paths), desc="Running sims...", total=len(image_paths)):

            

        #     # run simulation
        #     model = Dataset()
        #     model.run_sim_from_images(
        #         image_dir=image_paths,
        #         save_path=save_path,
        #         config=sim_config
        #     )
            
            

        