# imports
import jax.numpy as jnp
import os 
import glob

from turbencrypt.run_turbulence import Turbulence
from turbencrypt.compare_spectra import CompSpectra
from turbencrypt.make_forcing import Forcings, FourierTransform

sim_config = {
        'viscosity': 1e-3,
        'max_velocity': 2.0,
        'final_time': 25,
        'outer_steps': 25,
        'gridsize': 32,
        'max_courant_num': 0.001
    }

image_dir = "raw_images"
image_paths = glob.glob(f"{image_dir}/*")
fftmodel = FourierTransform()
model = Turbulence(**sim_config)
spectra = CompSpectra()

correlation_list = []
for image_path in image_paths:
    img = fftmodel.load_image(image_path)
    forcing_fn = lambda grid: Forcings().mod_kolmogorov_forcing(img, grid)
    simulation_at_t, forcing_array_x, forcing_array_y = model.run_turbulence(forcing_fn)
    corr = spectra.compare_spectra(simulation_at_t, forcing_array_x)
    correlation_list.append(corr)
avg_corr = jnp.mean(jnp.array(correlation_list))
print(avg_corr)