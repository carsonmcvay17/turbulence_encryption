# all the imports
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
from make_forcing import Forcings

class Turbulence():
    """
    Runs a simulation of turbulence with the modified Kolmogorov forcing
    """
    def run_turbulence(self, forcing_fn):
        viscosity = 1e-2
        max_velocity = 7
        grid = grids.Grid((256, 256), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
        dt = cfd.equations.stable_time_step(max_velocity, .1, viscosity, grid)
        
        # Change to ForcedNavierStokes2D2 with proper forcing_fn
        step_fn = spectral.time_stepping.crank_nicolson_rk4(
            navier_stokes.ForcedNavierStokes2D2(viscosity, grid, smooth=True, forcing_fn=forcing_fn),
            dt,
        )
        
        # run the simulation
        final_time = 25.0
        outer_steps = 10
        inner_steps = (final_time // dt) // 10
        trajectory_fn = cfd.funcutils.trajectory(
            cfd.funcutils.repeated(step_fn, inner_steps), outer_steps
        )

        v0 = cfd.initial_conditions.filtered_velocity_field(jax.random.PRNGKey(42), grid, max_velocity, 4)
        vorticity0 = cfd.finite_differences.curl_2d(v0).data
        vorticity_hat0 = jnp.fft.rfftn(vorticity0)

        _, trajectory = trajectory_fn(vorticity_hat0)

        spatial_coord = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0] # same for x and y
        coords = {
            'time': dt * jnp.arange(outer_steps) * inner_steps,
            'x': spatial_coord,
            'y': spatial_coord,
        }
        
        timestep_index = jnp.abs(coords['time']-0).argmin()
        simulation_at_t15 = trajectory[timestep_index]

        # Evaluate the pre-initialized forcing function with the grid variables
        vx, vy = v0  # Assuming `v0` provides x and y velocity components
        vx_grid_var = navier_stokes._get_grid_variable(vx.data, grid)
        vy_grid_var = navier_stokes._get_grid_variable(vy.data, grid)
        
        forcing_fn_with_grid = forcing_fn(grid)  # Use the passed forcing_fn correctly
        forcing_x, forcing_y = forcing_fn_with_grid((vx_grid_var, vy_grid_var))

        # Extract the data attributes of GridVariables
        forcing_array_x = forcing_x.data  # This is a JAX array
        forcing_array_y = forcing_y.data  # This is a JAX array

        return simulation_at_t15, forcing_array_x, forcing_array_y