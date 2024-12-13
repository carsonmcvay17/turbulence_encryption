import dataclasses
from typing import Callable, Optional

import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import forcings
from jax_cfd.base import grids
from jax_cfd.spectral import forcings as spectral_forcings
from jax_cfd.spectral import time_stepping
from jax_cfd.spectral import types
from jax_cfd.spectral import utils as spectral_utils
import make_forcing
from make_forcing import Forcings


TimeDependentForcingFn = Callable[[float], types.Array]
RandomSeed = int
ForcingModule = Callable[[grids.Grid, RandomSeed], TimeDependentForcingFn]





def _get_grid_variable(arr,
                       grid,
                       bc=boundaries.periodic_boundary_conditions(2),
                       offset=(0.5, 0.5)):
  return grids.GridVariable(grids.GridArray(arr, offset, grid), bc)

@dataclasses.dataclass
class NavierStokes2D2(time_stepping.ImplicitExplicitODE):
  """Breaks the Navier-Stokes equation into implicit and explicit parts.

  Implicit parts are the linear terms and explicit parts are the non-linear
  terms.

  Attributes:
    viscosity: strength of the diffusion term
    grid: underlying grid of the process
    smooth: smooth the advection term using the 2/3-rule.
    forcing_fn: forcing function, if None then no forcing is used.
    drag: strength of the drag. Set to zero for no drag.
  """
  viscosity: float
  grid: grids.Grid
  drag: float = 0.
  smooth: bool = True
  forcing_fn: Optional[Callable[[grids.Grid], forcings.ForcingFn]] = None
  _forcing_fn_with_grid = None

  def __post_init__(self):
    self.kx, self.ky = self.grid.rfft_mesh()
    self.laplace = (jnp.pi * 2j)**2 * (self.kx**2 + self.ky**2)
    self.filter_ = spectral_utils.brick_wall_filter_2d(self.grid)
    self.linear_term = self.viscosity * self.laplace - self.drag

    # setup the forcing function with the caller-specified grid.
    if self.forcing_fn is not None:
      self._forcing_fn_with_grid = self.forcing_fn(self.grid)

  def explicit_terms(self, vorticity_hat):
    velocity_solve = spectral_utils.vorticity_to_velocity(self.grid)
    vxhat, vyhat = velocity_solve(vorticity_hat)
    vx, vy = jnp.fft.irfftn(vxhat), jnp.fft.irfftn(vyhat)

    grad_x_hat = 2j * jnp.pi * self.kx * vorticity_hat
    grad_y_hat = 2j * jnp.pi * self.ky * vorticity_hat
    grad_x, grad_y = jnp.fft.irfftn(grad_x_hat), jnp.fft.irfftn(grad_y_hat)

    advection = -(grad_x * vx + grad_y * vy)
    advection_hat = jnp.fft.rfftn(advection)

    if self.smooth is not None:
      advection_hat *= self.filter_

    terms = advection_hat

    if self.forcing_fn is not None:
      fx, fy = self._forcing_fn_with_grid((_get_grid_variable(vx, self.grid),
                                           _get_grid_variable(vy, self.grid)))
      fx_hat, fy_hat = jnp.fft.rfft2(fx.data), jnp.fft.rfft2(fy.data)
      terms += spectral_utils.spectral_curl_2d((self.kx, self.ky),
                                               (fx_hat, fy_hat))

    return terms

  def implicit_terms(self, vorticity_hat):
    return self.linear_term * vorticity_hat

  def implicit_solve(self, vorticity_hat, time_step):
    return 1 / (1 - time_step * self.linear_term) * vorticity_hat


# pylint: disable=g-doc-args,g-doc-return-or-yield,invalid-name
def ForcedNavierStokes2D2(viscosity, grid, smooth, forcing_fn):
  """Sets up the flow that is used in Kochkov et al. [1].

  The authors of [1] based their work on Boffetta et al. [2].

  References:
    [1] Machine learning–accelerated computational fluid dynamics. Dmitrii
    Kochkov, Jamie A. Smith, Ayya Alieva, Qing Wang, Michael P. Brenner, Stephan
    Hoyer Proceedings of the National Academy of Sciences May 2021, 118 (21)
    e2101784118; DOI: 10.1073/pnas.2101784118.
    https://doi.org/10.1073/pnas.2101784118

    [2] Boffetta, Guido, and Robert E. Ecke. "Two-dimensional turbulence."
    Annual review of fluid mechanics 44 (2012): 427-451.
    https://doi.org/10.1146/annurev-fluid-120710-101240
  """
  wave_number = 1
  offsets = ((0, 0), (0, 0))
  # pylint: disable=g-long-lambda
  # forcing_fn = lambda grid: forcings.kolmogorov_forcing(
  #     grid, k=wave_number, offsets=offsets)
  # commenting out all forcing functions 
  # forcing_fn = lambda grid: make_forcing.Forcings.mod_kolmogorov_forcing(
  #     grid, k=wave_number, offsets=offsets)
  # trying different forcing functions to see if it breaks
  # forcing_fn= lambda grid: forcings.taylor_green_forcing(grid, k=wave_number)
  # forcing_fn = lambda grid: forcings.no_forcing(grid)
  return NavierStokes2D2(
    viscosity,
    grid,
    drag = 0.1,
    smooth = smooth,
    forcing_fn=forcing_fn
  )
 