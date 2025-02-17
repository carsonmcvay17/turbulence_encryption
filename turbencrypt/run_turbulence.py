# all the imports
import jax
import jax.numpy as jnp

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral

from dataclasses import dataclass
import turbencrypt.navier_stokes as navier_stokes
from turbencrypt.navier_stokes import NavierStokes2D2
from turbencrypt.make_forcing import Forcings
from turbencrypt.initial_conditions import quiescient

@dataclass
class Turbulence:
    """
    Runs a simulation of turbulence with the modified Kolmogorov forcing
    parameters:
    viscosity-changes the viscosity of the simulation
    max_velocity-changes the max_velocity of the simulation
    final_time-changes the final time
    max_courant_num-adjusts the time step, smaller makes the step bigger
    outer steps-?
    gridsize-grid size of simulations
    peak_wavenum-?
    random_seed-controls random seed
    """
    viscosity: float = 1e-2
    max_velocity: float = 7.0

    final_time: float = 25.0
    max_courant_num: float = 0.1
    outer_steps: int = 10

    gridsize: int = 256
    peak_wavenum: int = 4

    random_seed: int = 42


    def run_turbulence(self, forcing_fn):
        grid = grids.Grid((self.gridsize, self.gridsize), domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))
        dt = cfd.equations.stable_time_step(self.max_velocity, self.max_courant_num, self.viscosity, grid)
        
        # Change to ForcedNavierStokes2D2 with proper forcing_fn
        step_fn = spectral.time_stepping.crank_nicolson_rk4(
            navier_stokes.ForcedNavierStokes2D2(self.viscosity, grid, smooth=True, forcing_fn=forcing_fn),
            dt,
        )
        
        # run the simulation
        inner_steps = (self.final_time // dt) // self.outer_steps
        trajectory_fn = cfd.funcutils.trajectory(
            cfd.funcutils.repeated(step_fn, inner_steps), self.outer_steps
        )

        v0 = quiescient(jax.random.PRNGKey(self.random_seed), grid, self.max_velocity, self.peak_wavenum)
        vorticity0 = cfd.finite_differences.curl_2d(v0).data
        vorticity_hat0 = jnp.fft.rfftn(vorticity0)

        _, trajectory = trajectory_fn(vorticity_hat0)

        spatial_coord = jnp.arange(grid.shape[0]) * 2 * jnp.pi / grid.shape[0] # same for x and y
        coords = {
            'time': dt * jnp.arange(self.outer_steps) * inner_steps,
            'x': spatial_coord,
            'y': spatial_coord,
        }

        transformed_traj = jnp.fft.irfftn(trajectory, axes=(1,2))
        
        timestep_index = jnp.abs(coords['time']-0).argmin()
        simulation_at_t = transformed_traj[timestep_index]
       

        # Evaluate the pre-initialized forcing function with the grid variables
        vx, vy = v0  # Assuming `v0` provides x and y velocity components
        vx_grid_var = navier_stokes._get_grid_variable(vx.data, grid)
        vy_grid_var = navier_stokes._get_grid_variable(vy.data, grid)
        
        forcing_fn_with_grid = forcing_fn(grid)  # Use the passed forcing_fn correctly
        forcing_x, forcing_y = forcing_fn_with_grid((vx_grid_var, vy_grid_var))

        # Extract the data attributes of GridVariables
        forcing_array_x = forcing_x.data  # This is a JAX array
        forcing_array_y = forcing_y.data  # This is a JAX array

        return simulation_at_t, forcing_array_x, forcing_array_y