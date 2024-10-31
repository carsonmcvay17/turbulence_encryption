import jax_cfd
import matplotlib.image as mpi
import matplotlib.pyplot as plt
import numpy as np
import os 
import math

import functools
from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from jax_cfd.base import equations
from jax_cfd.base import filter_utils
from jax_cfd.base import grids
from jax_cfd.base import validation_problems

Array = grids.Array
GridArrayVector = grids.GridArrayVector
GridVariableVector = grids.GridVariableVector
ForcingFn = Callable[[GridVariableVector], GridArrayVector]


class Forcings:
    """
    A class for creating a forcing function and applying it in a grid
    """
    def __init__(self, alpha=0):
        # idk what we need here yet
        self.alpha = alpha

    def kolmogorov_focing(
            grid: grids.Grid, 
            scale: float = 1, 
            k: int = 2, 
            swap_xy: bool = False, 
            offsets: Optional[Tuple[Tuple[float, ...], ...]] = None, 
    )   -> ForcingFn:
        """Returns the Kolmogorov focing function for turbulence in 2D"""
        if offsets is None:
            offsets = grid.cell_faces

        if swap_xy:
            x = grid.mesh(offsets[1])[0]
            v = scale * grids.GridArray(jnp.sin(k*x), offsets[1], grid)

            if grid.ndim == 2:
                u = grids.GridArray(jnp.zeros_like(v.data), (1, 1/2), grid)
                f = (u, v)
            elif grid.ndim == 3:
                u = grids. GridArray(jnp.zeros_like(v.data), (1, 1/2, 1/2), grid)
                w = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1/2, 1), grid)
                f = (u, v, w)
            else:
                raise NotImplementedError
        else:
            y = grid.mesh(offsets[0])[1]
            u = scale * grids.GridArray(jnp.sin(k*y), offsets[0], grid)

            if grid.ndim == 2:
                v = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1), grid)
                f = (u,v)
            elif grid.ndim == 3:
                v = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1, 1/2), grid)
                w = grids.GridArray(jnp.zeros_like(u.data), (1/2, 1/2, 1), grid)
                f = (u, v, w)
            else: 
                raise NotImplementedError
        
        def forcing(v):
            del v
            return f 
        return forcing
              
                


 
    
        

