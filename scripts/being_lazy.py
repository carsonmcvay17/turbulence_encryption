import os
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import xarray
import jax_cfd as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
from jax_cfd.spectral import equations
from jax_cfd.base import equations as beq
from jax_cfd.spectral import time_stepping
from jax_cfd.base import funcutils
from jax_cfd.base import initial_conditions
from jax_cfd.base import finite_differences
import jax.numpy as jnp

# Create directory to save the images
output_dir = f"movie_files/"
os.makedirs(output_dir, exist_ok=True)

# physical parameters
viscosity = 1e-3
max_velocity = 7
grid = grids.Grid((256, 256), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
dt = beq.stable_time_step(max_velocity, .5, viscosity, grid)

# setup step function using crank-nicolson runge-kutta order 4
smooth = True # use anti-aliasing 

# **use predefined settings for Kolmogorov flow**
step_fn = time_stepping.crank_nicolson_rk4(
    equations.ForcedNavierStokes2D(viscosity, grid, smooth=smooth), dt)

# run the simulation up until time 25.0 but only save 10 frames for visualization
final_time = 25.0
outer_steps = 10
inner_steps = (final_time // dt) // 10

trajectory_fn = funcutils.trajectory(
    funcutils.repeated(step_fn, inner_steps), outer_steps)

# create an initial velocity field and compute the fft of the vorticity
v0 = initial_conditions.filtered_velocity_field(jax.random.PRNGKey(42), grid, max_velocity, 4)
vorticity0 = finite_differences.curl_2d(v0).data
vorticity_hat0 = jnp.fft.rfftn(vorticity0)

_, trajectory = trajectory_fn(vorticity_hat0)

# transform the trajectory into real-space and wrap in xarray for plotting
spatial_coord = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0] # same for x and y
coords = {
  'time': dt * jnp.arange(outer_steps) * inner_steps,
  'x': spatial_coord,
  'y': spatial_coord,
}

# Plot and save images at each time step
for i, time_step in enumerate(trajectory):
    # Convert the Fourier transformed data into real space
    real_space_data = jnp.fft.irfftn(time_step, axes=(1, 2))
    
    # Create a plot for the current time step
    fig, ax = plt.subplots(figsize=(8, 6))
    xarray.DataArray(
        real_space_data,
        dims=["time", "x", "y"],
        coords=coords
    ).plot.imshow(
        col='time', col_wrap=5, cmap=sns.cm.icefire, robust=True, ax=ax
    )
    
    # Save the current plot as an image
    image_filename = os.path.join(output_dir, f"my_fig{i:04d}.png")
    plt.savefig(image_filename)
    plt.close(fig)